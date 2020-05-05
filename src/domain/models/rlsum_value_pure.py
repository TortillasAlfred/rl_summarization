from src.domain.environment import BanditSummarizationEnvironment
import src.domain.mcts as mcts

import pytorch_lightning as pl
import logging
import torch
from torch.utils.data import Subset, DataLoader
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
import numpy as np
from torchtext.data import BucketIterator
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import torch.multiprocessing as mp
import os
import functools


class RLSumValuePure(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super(RLSumValuePure, self).__init__()
        self.hparams = hparams
        self.dataset = dataset
        self.environment = BanditSummarizationEnvironment(
            reward, episode_length=hparams.n_sents_per_summary
        )

        self.embedding_dim = self.dataset.embedding_dim
        self.pad_idx = self.dataset.pad_idx
        self.splits = self.dataset.get_splits()
        self.n_epochs_done = 0

        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size
        self.hidden_dim = hparams.hidden_dim
        self.decoder_dim = hparams.decoder_dim
        self.n_repeats_per_sample = hparams.n_repeats_per_sample
        self.learning_rate = hparams.learning_rate
        self.epsilon = hparams.epsilon
        self.dirichlet_epsilon = hparams.dirichlet_epsilon
        self.n_sents_per_summary = hparams.n_sents_per_summary
        self.c_puct = hparams.c_puct
        self.n_mcts_samples = hparams.n_mcts_samples
        self.dropout = hparams.dropout
        self.weight_decay = hparams.weight_decay

        self.__build_model(hparams.hidden_dim)
        self.model = RLSummModel(hparams.hidden_dim, hparams.decoder_dim, self.dropout,)

        mp.set_start_method("forkserver", force=True)
        if hparams.n_jobs_for_mcts == -1:
            n_processes = os.cpu_count()
        else:
            n_processes = hparams.n_jobs_for_mcts
        self.pool = mp.Pool(processes=n_processes)

    def __build_model(self, hidden_dim):
        self.embeddings = torch.nn.Embedding.from_pretrained(
            self.dataset.vocab.vectors, freeze=False, padding_idx=self.pad_idx
        )
        self.wl_encoder = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout,
        )

    def word_level_encoding(self, contents):
        valid_tokens = ~(contents == self.pad_idx)
        sentences_len = valid_tokens.sum(dim=-1)
        valid_sentences = sentences_len > 0
        contents = self.embeddings(contents)
        orig_shape = contents.shape
        contents = self.wl_encoder(contents.view(-1, *orig_shape[2:]))[0].reshape(
            *orig_shape[:3], -1
        )
        contents = contents * valid_tokens.unsqueeze(-1)
        contents = contents.sum(-2)
        word_level_encodings = torch.zeros_like(contents)
        word_level_encodings[valid_sentences] = contents[
            valid_sentences
        ] / sentences_len[valid_sentences].unsqueeze(-1)
        return word_level_encodings, valid_sentences

    def __extract_features(self, contents):
        contents, valid_sentences = self.word_level_encoding(contents)
        sent_contents, doc_contents = self.model.sentence_level_encoding(contents)

        return sent_contents, doc_contents, valid_sentences

    def mcts(self, sent_contents, doc_contents, states, valid_sentences):
        mcts_pures = self.pool.map(
            mcts.RLSumValuePureProcess(
                n_samples=self.n_mcts_samples,
                c_puct=self.c_puct,
                n_sents_per_summary=self.n_sents_per_summary,
            ),
            zip(
                states,
                valid_sentences,
                [s.scores for s in self.environment.reward_scorers],
            ),
        )

        return (
            [m_i[0] for m in mcts_pures for m_i in m],
            torch.stack([torch.stack([m_i[1] for m_i in m]) for m in mcts_pures]),
        )

    def forward(self, batch, subset):
        torch.set_grad_enabled(False)
        states = self.environment.init(batch, subset)

        sent_contents, doc_contents, valid_sentences = self.__extract_features(
            batch.content
        )

        if subset == "train":
            prediction_states, mcts_vals = self.mcts(
                sent_contents, doc_contents, states, valid_sentences
            )

            for _ in range(self.n_sents_per_summary):
                (
                    action_dist,
                    valid_sentences,
                    action_vals,
                ) = self.model.produce_affinities(
                    sent_contents, doc_contents, states, valid_sentences
                )
                _, greedy_idxs = action_vals.mean(-1).max(dim=-1)
                states, greedy_rewards = self.environment.update(
                    greedy_idxs, action_dist
                )

            self.environment.soft_init(batch, subset)

            torch.set_grad_enabled(True)

            sent_contents, doc_contents, valid_sentences = self.__extract_features(
                batch.content
            )
            sent_contents = torch.repeat_interleave(
                sent_contents, int(len(prediction_states) / batch.batch_size), dim=0
            )
            doc_contents = torch.repeat_interleave(
                doc_contents, int(len(prediction_states) / batch.batch_size), dim=0
            )
            valid_sentences = torch.repeat_interleave(
                valid_sentences, int(len(prediction_states) / batch.batch_size), dim=0
            )
            _, available_sents, action_vals = self.model.produce_affinities(
                sent_contents, doc_contents, prediction_states, valid_sentences
            )

            mcts_vals = torch.cat([m for m in mcts_vals], dim=0)
            loss = (
                (mcts_vals.to(valid_sentences.device) - action_vals) ** 2
            ) / batch.batch_size
            loss = loss.sum()

            return greedy_rewards, loss
        else:
            for _ in range(self.n_sents_per_summary):
                (
                    action_dist,
                    valid_sentences,
                    action_vals,
                ) = self.model.produce_affinities(
                    sent_contents, doc_contents, states, valid_sentences
                )
                _, greedy_idxs = action_dist.probs.max(dim=-1)
                states, greedy_rewards = self.environment.update(
                    greedy_idxs, action_dist
                )
            return greedy_rewards

    def get_step_output(self, **kwargs):
        output_dict = {}

        log_dict = self.environment.get_logged_metrics()
        log_dict = {k: torch.tensor(v).mean() for k, v in log_dict.items()}
        for key, value in kwargs.items():
            log_dict[key] = value
        output_dict["log"] = log_dict

        if "loss" in log_dict:
            output_dict["loss"] = log_dict["loss"]

        tqdm_keys = ["rouge_mean", "greedy_reward"]
        output_dict["progress_bar"] = {k: log_dict[k] for k in tqdm_keys}

        return output_dict

    def training_step(self, batch, batch_idx):
        greedy_rewards, loss = self.forward(batch, subset="train")

        return self.get_step_output(loss=loss, greedy_reward=greedy_rewards.mean())

    def validation_step(self, batch, batch_idx):
        greedy_rewards = self.forward(batch, subset="val").mean(0)

        reward_dict = {
            "val_greedy_rouge_1": greedy_rewards[0],
            "val_greedy_rouge_2": greedy_rewards[1],
            "val_greedy_rouge_L": greedy_rewards[2],
            "val_greedy_rouge_mean": greedy_rewards.mean(),
        }

        return reward_dict

    def validation_epoch_end(self, outputs):
        output_dict = self.generic_epoch_end(outputs)

        self.lr_scheduler.step(output_dict["log"]["val_greedy_rouge_mean"])
        output_dict["log"]["learning_rate"] = self.trainer.optimizers[0].param_groups[
            1
        ]["lr"]

        return output_dict

    def test_step(self, batch, batch_idx):
        greedy_rewards = self.forward(batch, subset="test").mean(0)

        reward_dict = {
            "test_greedy_rouge_1": greedy_rewards[0],
            "test_greedy_rouge_2": greedy_rewards[1],
            "test_greedy_rouge_L": greedy_rewards[2],
            "test_greedy_rouge_mean": greedy_rewards.mean(),
        }

        return reward_dict

    def generic_epoch_end(self, outputs, is_test=False):
        combined_outputs = {}
        log_dict = {}

        for key in outputs[0]:
            log_dict[key] = np.mean([output[key] for output in outputs])

        combined_outputs["log"] = log_dict

        if is_test:
            combined_outputs["progress_bar"] = log_dict
        else:
            tqdm_keys = ["rouge_mean"]
            combined_outputs["progress_bar"] = {
                k: v
                for k, v in log_dict.items()
                if any([t_k in k for t_k in tqdm_keys])
            }

        return combined_outputs

    def test_epoch_end(self, outputs):
        return self.generic_epoch_end(outputs, is_test=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.embeddings.parameters(),
                    "lr": self.learning_rate * 0.1,
                },
                {"params": self.wl_encoder.parameters()},
                {"params": self.model.parameters()},
            ],
            lr=self.learning_rate,
            betas=[0, 0.999],
            weight_decay=self.weight_decay,
        )

        self.lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.1, verbose=True
        )

        return optimizer

    def train_dataloader(self):
        return BucketIterator(
            self.splits["train"],
            train=True,
            batch_size=self.train_batch_size,
            sort=False,
            device=self.embeddings.weight.device,
        )

    def val_dataloader(self):
        return BucketIterator(
            self.splits["val"],
            train=False,
            batch_size=self.test_batch_size,
            sort=False,
            device=self.embeddings.weight.device,
        )

    def test_dataloader(self):
        return BucketIterator(
            self.splits["test"],
            train=False,
            batch_size=self.test_batch_size,
            sort=False,
            device=self.embeddings.weight.device,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return RLSumValuePure(dataset, reward, config,)


class RLSummModel(torch.nn.Module):
    def __init__(self, hidden_dim, decoder_dim, dropout):
        super().__init__()
        self.sl_encoder = torch.nn.LSTM(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 * 3, decoder_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, 3),
            torch.nn.Sigmoid(),
        )

    def sentence_level_encoding(self, contents):
        sent_contents, (doc_contents, _) = self.sl_encoder(contents)
        doc_contents = doc_contents.view(2, 2, *doc_contents.shape[-2:])
        doc_contents = (
            torch.cat([d_i for d_i in doc_contents], dim=-1)
            .mean(0, keepdim=True)
            .permute(1, 0, 2)
        )

        return sent_contents, doc_contents

    def decoding(self, contents):
        return self.decoder(contents).squeeze(-1)

    def get_summ_sents_from_states(self, sent_contents, states):
        summ_sents = []

        for article_sents, state in zip(sent_contents, states):
            if len(state.summary_idxs) == 0:
                summ_sents.append(torch.zeros_like(sent_contents[0][0]).unsqueeze(0))
            else:
                summ_sents_i = [article_sents[i] for i in state.summary_idxs]
                summ_sents_i = torch.stack(summ_sents_i)
                summ_sents.append(summ_sents_i)

        return torch.nn.utils.rnn.pad_sequence(summ_sents, batch_first=True)

    def produce_affinities(self, sent_contents, doc_contents, states, valid_sentences):
        n_sents = sent_contents.shape[1]
        summ_contents = self.get_summ_sents_from_states(sent_contents, states)
        _, summ_contents = self.sentence_level_encoding(summ_contents)

        summ_contents = torch.cat(n_sents * [summ_contents], dim=1)
        doc_contents = torch.cat(n_sents * [doc_contents], dim=1)
        contents = self.decoding(
            torch.cat([doc_contents, summ_contents, sent_contents], dim=-1)
        )

        available_sents = torch.ones_like(valid_sentences)
        for state, a_s in zip(states, available_sents):
            for idx in state.summary_idxs:
                a_s[idx] = False

        valid_sentences = valid_sentences & available_sents
        contents = contents * valid_sentences.unsqueeze(-1)

        probs = contents.mean(-1)
        probs[probs > 0] = (math.log(10) * probs[probs > 0]).exp()

        return Categorical(probs=probs), valid_sentences, contents

    def forward(self, x):
        pass
