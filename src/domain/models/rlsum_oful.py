from src.domain.environment import BanditSummarizationEnvironment
import src.domain.mcts_oful as mcts_oful

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
from itertools import combinations


class RLSumOFUL(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super(RLSumOFUL, self).__init__()
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
        self.lambda_oful = hparams.lambda_oful
        self.S = hparams.S
        self.R = hparams.R
        self.delta = hparams.delta
        self.D_t_source = hparams.D_t_source
        self.warmup_batches = hparams.warmup_batches
        self.batch_idx = 0
        self.alpha_oful = hparams.alpha_oful

        self.__build_model(hparams.hidden_dim)
        self.model = RLSummModel(hparams.hidden_dim, hparams.decoder_dim, self.dropout,)

        mp.set_start_method("forkserver", force=True)
        if hparams.n_jobs_for_mcts == -1:
            self.n_processes = os.cpu_count()
        else:
            self.n_processes = hparams.n_jobs_for_mcts
        self.pool = mp.Pool(processes=self.n_processes)

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

        return sent_contents, doc_contents, valid_sentences, contents

    def warmup_oful(self, valid_sentences):
        all_sampled_summs = []
        all_scores = []

        for valid_sents, scorer in zip(
            valid_sentences, self.environment.reward_scorers
        ):
            all_sums = list(combinations(list(range(valid_sents.sum())), 3))
            sampled_summs = np.random.choice(len(all_sums), 250, replace=True)
            sampled_summs = [all_sums[summ] for summ in sampled_summs]
            scores = torch.tensor(
                [scorer.scores[tuple(summ)] for summ in sampled_summs]
            ).to(valid_sentences.device)
            all_scores.append(scores)
            all_sampled_summs.append(sampled_summs)

        all_scores = torch.cat(all_scores)

        return all_sampled_summs, all_scores

    def mcts_oful(
        self, sent_contents, doc_contents, states, valid_sentences,
    ):
        device = sent_contents.device

        mcts_pures = self.pool.map(
            mcts_oful.RLSumOFULValueProcess(
                n_samples=self.n_mcts_samples,
                lambda_oful=self.lambda_oful,
                alpha_oful=self.alpha_oful,
                action_dim=sent_contents.shape[-1],
                device=device,
                n_sents_per_summary=self.n_sents_per_summary,
            ),
            zip(
                sent_contents.to(device),
                doc_contents.to(device),
                states,
                valid_sentences.to(device),
                [s.scores for s in self.environment.reward_scorers],
            ),
        )

        mcts_theta_hats = torch.stack([m[0] for m in mcts_pures]).transpose(2, 1)

        return mcts_theta_hats.cuda()

    def forward(self, batch, subset):
        torch.set_grad_enabled(False)
        states = self.environment.init(batch, subset)

        (
            sent_contents,
            doc_contents,
            valid_sentences,
            raw_sent_contents,
        ) = self.__extract_features(batch.content)

        if subset == "train":

            if self.batch_idx >= self.warmup_batches:
                mcts_theta_hats = self.mcts_oful(
                    sent_contents, doc_contents, states, valid_sentences,
                )

                all_selected_sents = []

                for theta_hat_doc, sent_conts, val_sents in zip(
                    mcts_theta_hats, sent_contents, valid_sentences
                ):
                    sent_predicted_vals = theta_hat_doc.mm(sent_conts.T)
                    sent_predicted_vals = sent_predicted_vals.squeeze()[val_sents]
                    _, selected_sents = sent_predicted_vals.topk(
                        self.n_sents_per_summary
                    )
                    all_selected_sents.append(selected_sents)

                for selected_sents_step in zip(*all_selected_sents):
                    _, mcts_rewards = self.environment.update(selected_sents_step)

                self.environment.soft_init(batch, subset)

                torch.set_grad_enabled(True)

                (
                    sent_contents,
                    doc_contents,
                    valid_sentences,
                    _,
                ) = self.__extract_features(batch.content)
                theta_hats = self.model.produce_theta_hat(doc_contents)

                all_selected_sents = []

                for theta_hat_doc, sent_conts, val_sents in zip(
                    theta_hats, sent_contents, valid_sentences
                ):
                    sent_predicted_vals = theta_hat_doc.mm(sent_conts.T)
                    sent_predicted_vals = sent_predicted_vals.squeeze()[val_sents]
                    _, selected_sents = sent_predicted_vals.topk(
                        self.n_sents_per_summary
                    )
                    all_selected_sents.append(selected_sents)

                for selected_sents_step in zip(*all_selected_sents):
                    _, greedy_rewards = self.environment.update(selected_sents_step)

                loss = (mcts_theta_hats.to(valid_sentences.device) - theta_hats) ** 2
                loss = loss.sum()

                return greedy_rewards, loss, mcts_rewards
            else:
                sampled_summs, sampled_scores = self.warmup_oful(valid_sentences)

                theta_hats = self.model.produce_theta_hat(doc_contents)

                all_selected_sents = []

                for theta_hat_doc, sent_conts, val_sents in zip(
                    theta_hats, sent_contents, valid_sentences
                ):
                    sent_predicted_vals = theta_hat_doc.mm(sent_conts.T)
                    sent_predicted_vals = sent_predicted_vals.squeeze()[val_sents]
                    _, selected_sents = sent_predicted_vals.topk(
                        self.n_sents_per_summary
                    )
                    all_selected_sents.append(selected_sents)

                for selected_sents_step in zip(*all_selected_sents):
                    _, greedy_rewards = self.environment.update(selected_sents_step)

                self.environment.soft_init(batch, subset)

                torch.set_grad_enabled(True)

                (
                    sent_contents,
                    doc_contents,
                    valid_sentences,
                    _,
                ) = self.__extract_features(batch.content)
                predicted_scores = self.model.pretraining_output(
                    sent_contents, doc_contents, sampled_summs
                )

                loss = (
                    sampled_scores.to(valid_sentences.device) - predicted_scores
                ) ** 2
                loss = loss.sum()

                return greedy_rewards, loss, np.zeros_like(greedy_rewards)
        else:
            theta_hats = self.model.produce_theta_hat(doc_contents)

            all_selected_sents = []

            for theta_hat_doc, sent_conts, val_sents in zip(
                theta_hats, sent_contents, valid_sentences
            ):
                sent_predicted_vals = theta_hat_doc.mm(sent_conts.T)
                sent_predicted_vals = sent_predicted_vals.squeeze()[val_sents]
                _, selected_sents = sent_predicted_vals.topk(self.n_sents_per_summary)
                all_selected_sents.append(selected_sents)

            for selected_sents_step in zip(*all_selected_sents):
                _, greedy_rewards = self.environment.update(selected_sents_step)

            return greedy_rewards

    def get_step_output(self, **kwargs):
        output_dict = {}

        log_dict = self.environment.get_logged_metrics()
        log_dict = {
            k: torch.tensor(v).mean() if not torch.is_tensor(v) else v.mean()
            for k, v in log_dict.items()
        }
        for key, value in kwargs.items():
            log_dict[key] = value
        output_dict["log"] = log_dict

        if "loss" in log_dict:
            output_dict["loss"] = log_dict["loss"]

        tqdm_keys = ["mcts_rewards", "greedy_reward"]
        output_dict["progress_bar"] = {k: log_dict[k] for k in tqdm_keys}

        return output_dict

    def training_step(self, batch, batch_idx):
        greedy_rewards, loss, mcts_rewards = self.forward(batch, subset="train")
        self.batch_idx += 1

        return self.get_step_output(
            loss=loss,
            greedy_reward=greedy_rewards.mean(),
            mcts_rewards=mcts_rewards.mean(),
        )

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

        if self.batch_idx >= self.warmup_batches:
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
        return RLSumOFUL(dataset, reward, config,)


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
            torch.nn.Linear(hidden_dim * 2 * 4, decoder_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, 3),
            torch.nn.Sigmoid(),
        )
        self.theta_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
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

    def get_sents_from_summs(self, sent_contents, sampled_summs):
        all_sents = []

        for sents_doc, sampled_sents in zip(sent_contents, sampled_summs):
            for summ in sampled_sents:
                all_sents.append(torch.cat([sents_doc[sent_id] for sent_id in summ]))

        return torch.stack(all_sents)

    def pretraining_output(self, sent_contents, doc_contents, sampled_summs):
        n_samples = len(sampled_summs[0])

        summ_contents = self.get_sents_from_summs(sent_contents, sampled_summs)
        doc_contents = torch.repeat_interleave(doc_contents, n_samples, dim=0).squeeze(
            1
        )
        predicted_scores = self.decoder(
            torch.cat([doc_contents, summ_contents], dim=-1)
        )

        return predicted_scores

    def produce_theta_hat(self, doc_contents):
        return self.theta_decoder(doc_contents)

    def forward(self, x):
        pass
