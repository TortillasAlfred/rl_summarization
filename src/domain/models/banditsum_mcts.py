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
from joblib import delayed, Parallel


class BanditSumMCTS(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super(BanditSumMCTS, self).__init__()
        self.hparams = hparams
        self.dataset = dataset
        self.environment = BanditSummarizationEnvironment(reward, episode_length=1)

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

        self.__build_model(hparams.hidden_dim, hparams.decoder_dim)

    def __build_model(self, hidden_dim, decoder_dim):
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
        self.sl_encoder = torch.nn.LSTM(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout,
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, decoder_dim),
            torch.nn.Dropout(self.dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, 1),
            torch.nn.Sigmoid(),
        )

    def __word_level_encoding(self, contents):
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

    def __sentence_level_encoding(self, contents):
        return self.sl_encoder(contents)[0]

    def __decoding(self, contents):
        return self.decoder(contents).squeeze(-1)

    def __produce_affinities(self, contents, states):
        contents, valid_sentences = self.__word_level_encoding(contents)
        contents = self.__sentence_level_encoding(contents)
        contents = self.__decoding(contents)
        contents = contents * valid_sentences

        return Categorical(probs=contents), valid_sentences

    def mcts(self, priors, valid_sentences):
        mcts_pures = Parallel(n_jobs=-1)(
            delayed(mcts.bs_mcts)(
                prior.cpu(),
                valid_sents.cpu(),
                self.n_mcts_samples,
                reward_scorer.scores,
                self.c_puct,
                self.n_sents_per_summary,
                self.dirichlet_epsilon,
            )
            for prior, valid_sents, reward_scorer in zip(
                priors, valid_sentences, self.environment.reward_scorers
            )
        )

        return torch.stack(mcts_pures)

    def forward(self, batch, subset):
        states = self.environment.init(batch, subset)
        action_dist, valid_sentences = self.__produce_affinities(batch.content, states)
        greedy_idxs = action_dist.probs.argsort(descending=True)[
            :, : self.n_sents_per_summary
        ]
        _, greedy_rewards = self.environment.update(greedy_idxs, action_dist)

        if subset == "train":
            _ = self.environment.soft_init(batch, subset, n_repeats=1)

            mcts_probs = self.mcts(action_dist.probs, valid_sentences)

            mcts_idxs = mcts_probs.argsort(descending=True)[
                :, : self.n_sents_per_summary
            ]

            _, mcts_rewards = self.environment.update(
                mcts_idxs, Categorical(probs=mcts_probs), is_mcts=True
            )

            loss = (-mcts_probs.to(valid_sentences.device) * action_dist.logits).sum()

            return mcts_rewards, greedy_rewards, loss
        else:
            return greedy_rewards

    def get_step_output(self, **kwargs):
        output_dict = {}

        log_dict = vars(self.environment.logged_metrics)
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
        generated_rewards, greedy_rewards, loss = self.forward(batch, subset="train")

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
                {"params": self.sl_encoder.parameters()},
                {"params": self.decoder.parameters()},
            ],
            lr=self.learning_rate,
            betas=[0, 0.999],
            weight_decay=self.weight_decay,
        )

        self.lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.2, verbose=True
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
        return BanditSumMCTS(dataset, reward, config,)
