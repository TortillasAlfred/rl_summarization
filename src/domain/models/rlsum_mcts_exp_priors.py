from src.domain.loader_utils import TextDataCollator

import threading
import pickle

import pytorch_lightning as pl
import logging
import torch
from torch.utils.data import Subset, DataLoader
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
import numpy as np
from torchtext.data import BucketIterator, Batch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import torch.multiprocessing as mp
import os
import functools
from itertools import combinations
from collections import defaultdict, namedtuple
from joblib import Parallel, delayed

np.seterr(invalid="ignore")


class RLSumMCTSEXPPriors(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super().__init__()
        self.fields = dataset.fields
        self.pad_idx = dataset.pad_idx
        self.reward_builder = reward

        self.embedding_dim = dataset.embedding_dim
        self.pad_idx = dataset.pad_idx
        self.splits = dataset.get_splits()
        self.n_epochs_done = 0

        self.train_batch_size = hparams.train_batch_size
        self.num_workers = hparams.num_workers
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

        self.__build_model(hparams.hidden_dim, dataset)
        self.model = RLSummModel(
            hparams.hidden_dim,
            hparams.decoder_dim,
            self.dropout,
        )
        self.raw_run_done = False

        self.mcts_log_path = "/project/def-adurand/magod/rl_summ/mcts_exp_priors/"
        os.makedirs(self.mcts_log_path, exist_ok=True)

    def __build_model(self, hidden_dim, dataset):
        self.embeddings = torch.nn.Embedding.from_pretrained(
            dataset.vocab.vectors, freeze=False, padding_idx=self.pad_idx
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
        contents = self.wl_encoder(contents.view(-1, *orig_shape[2:]))[0].reshape(*orig_shape[:3], -1)
        contents = contents * valid_tokens.unsqueeze(-1)
        contents = contents.sum(-2)
        word_level_encodings = torch.zeros_like(contents)
        word_level_encodings[valid_sentences] = contents[valid_sentences] / sentences_len[valid_sentences].unsqueeze(-1)
        return word_level_encodings, valid_sentences

    def __extract_features(self, contents):
        contents, valid_sentences = self.word_level_encoding(contents)
        sent_contents, doc_contents = self.model.sentence_level_encoding(contents)

        return sent_contents, doc_contents, valid_sentences, contents

    def __get_reward_scorers(self, ids, subset):
        if subset in ["train", "val", "test"]:
            return [self.reward_builder.init_scorer(id, subset) for id in ids]
        else:
            raise ValueError(f'Bad subset : {subset}. Should be one of ["train", "val", "test].')

    def mcts_exp(self, scorers, ids, c_pucts):
        return Parallel(n_jobs=-1, verbose=1, backend="loky")(
            collect_sims(scorer, id, c_pucts, 20) for scorer, id in zip(scorers, ids)
        )

    def forward(self, batch, subset):
        raw_contents, contents, raw_abstracts, abstracts, ids, scorers = batch
        torch.set_grad_enabled(False)

        c_pucts = np.logspace(-2, 2, 5)

        results = self.mcts_exp(scorers, ids, c_pucts)
        keys = [key for r in results for key in r[0]]
        q_vals_hats = [vals for r in results for vals in r[1]]

        return (keys, q_vals_hats)

    def get_step_output(self, loss, greedy_rewards, mcts_rewards, max_scores):
        output_dict = {}

        log_dict = {
            "greedy_rouge_1": greedy_rewards[:, 0],
            "greedy_rouge_2": greedy_rewards[:, 1],
            "greedy_rouge_L": greedy_rewards[:, 2],
            "greedy_rouge_mean": greedy_rewards.mean(-1),
            "mcts_rouge_1": mcts_rewards[:, 0],
            "mcts_rouge_2": mcts_rewards[:, 1],
            "mcts_rouge_L": mcts_rewards[:, 2],
            "mcts_rouge_mean": mcts_rewards.mean(-1),
            "max_rouge_1": max_scores[:, 0],
            "max_rouge_2": max_scores[:, 1],
            "max_rouge_L": max_scores[:, 2],
            "max_rouge_mean": max_scores.mean(-1),
        }
        log_dict["loss"] = loss

        output_dict["log"] = log_dict

        if "loss" in log_dict:
            output_dict["loss"] = log_dict["loss"]

        tqdm_keys = ["mcts_rouge", "greedy_rouge", "max_rouge"]
        output_dict["progress_bar"] = {k: log_dict[f"{k}_mean"] for k in tqdm_keys}

        return output_dict

    def training_step(self, batch, batch_idx):
        greedy_rewards, loss, mcts_rewards, max_scores = self.forward(batch, subset="train")

        return self.get_step_output(
            loss=loss.to(self.device),
            greedy_rewards=greedy_rewards.to(self.device),
            mcts_rewards=mcts_rewards.to(self.device),
            max_scores=max_scores.to(self.device),
        )

    def training_step_end(self, outputs):
        self.batch_idx += 1

        all_logs = {}
        out_dict = {}

        for key, vals in outputs["log"].items():
            all_logs[key] = vals.mean()

        out_dict["log"] = all_logs

        out_dict["loss"] = all_logs["loss"]

        tqdm_keys = ["mcts_rouge", "greedy_rouge", "max_rouge"]
        out_dict["progress_bar"] = {k: all_logs[f"{k}_mean"] for k in tqdm_keys}

        return out_dict

    def validation_step(self, batch, batch_idx):
        greedy_rewards = self.forward(batch, subset="val")

        reward_dict = {
            "val_greedy_rouge_1": greedy_rewards[:, 0],
            "val_greedy_rouge_2": greedy_rewards[:, 1],
            "val_greedy_rouge_L": greedy_rewards[:, 2],
            "val_greedy_rouge_mean": greedy_rewards.mean(-1),
        }

        return reward_dict

    def validation_step_end(self, outputs):
        for vals in outputs.values():
            vals = vals.mean()

        return outputs

    def validation_epoch_end(self, outputs):
        output_dict = self.generic_epoch_end(outputs)

        if self.batch_idx >= self.warmup_batches:
            self.lr_scheduler.step(output_dict["log"]["val_greedy_rouge_mean"])

        output_dict["log"]["learning_rate"] = self.trainer.optimizers[0].param_groups[1]["lr"]

        return output_dict

    def test_step(self, batch, batch_idx):
        keys, q_vals_hats = self.forward(batch, subset="test")

        d = {}

        for key, q_vals in zip(keys, q_vals_hats):
            d[key] = q_vals

        with open(os.path.join(self.mcts_log_path, f"results_{batch_idx}.pck"), "wb") as f:
            pickle.dump(d, f)

    def test_step_end(self, outputs):
        pass

    def generic_epoch_end(self, outputs, is_test=False):
        combined_outputs = {}
        log_dict = {}

        for key in outputs[0]:
            log_dict[key] = torch.stack([output[key] for output in outputs]).mean()

        combined_outputs["log"] = log_dict

        if is_test:
            combined_outputs["progress_bar"] = log_dict
        else:
            tqdm_keys = ["rouge_mean"]
            combined_outputs["progress_bar"] = {
                k: v for k, v in log_dict.items() if any([t_k in k for t_k in tqdm_keys])
            }

        return combined_outputs

    def test_epoch_end(self, outputs):
        all_dict_paths = os.listdir(self.mcts_log_path)
        d = {}

        for path in all_dict_paths:
            with open(os.path.join(self.mcts_log_path, path), "rb") as f:
                d_i = pickle.load(f)

            for k, v in d_i.items():
                d[k] = v

        with open(os.path.join(self.mcts_log_path, "results.pck"), "wb") as f:
            pickle.dump(d, f)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.embeddings.parameters(),
                    "lr": self.learning_rate * 0.1,
                },
                {"params": self.wl_encoder.parameters()},
                {"params": self.model.theta_decoder.parameters()},
                {"params": self.model.sl_encoder.parameters()},
                {
                    "params": self.model.decoder.parameters(),
                    "lr": self.learning_rate * 0.1,
                },
            ],
            lr=self.learning_rate,
            betas=[0, 0.999],
            weight_decay=self.weight_decay,
        )

        self.lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.1, verbose=True)

        return optimizer

    def train_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.fields, self.reward_builder, subset="train"),
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.fields, self.reward_builder, subset="train"),
            batch_size=self.test_batch_size,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.fields, self.reward_builder, subset="train"),
            batch_size=self.test_batch_size,
            drop_last=True,
            num_workers=self.num_workers,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return RLSumMCTSEXPPriors(
            dataset,
            reward,
            config,
        )


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
            torch.nn.Tanh(),
        )

    def sentence_level_encoding(self, contents):
        sent_contents, (doc_contents, _) = self.sl_encoder(contents)
        doc_contents = doc_contents.view(2, 2, *doc_contents.shape[-2:])
        doc_contents = torch.cat([d_i for d_i in doc_contents], dim=-1).mean(0, keepdim=True).permute(1, 0, 2)

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
        doc_contents = torch.repeat_interleave(doc_contents, n_samples, dim=0).squeeze(1)
        predicted_scores = self.decoder(torch.cat([doc_contents, summ_contents], dim=-1))

        return predicted_scores

    def produce_theta_hat(self, doc_contents):
        return self.theta_decoder(doc_contents)

    def forward(self, x):
        pass


@delayed
def collect_sims(scorer, id, c_pucts, n_samples):
    keys = []
    q_vals_hats = []

    n_sents = min(scorer.scores.shape[0], 50)
    max_rouge = scorer.scores.mean(-1).max()

    results = [do_one_sample(scorer, c_pucts, n_sents, prior_choice) for prior_choice in ["best", "med", "worst"]]

    for i, res in enumerate(results):
        for c_puct, prior_choice, r in res:
            for q_val_sims, prior_max_score, prior_max_proba, tau in r:
                if prior_max_score:
                    keys.append(
                        (
                            c_puct,
                            n_sents,
                            max_rouge,
                            prior_max_score,
                            prior_max_proba,
                            prior_choice,
                            tau,
                            (id, i),
                        )
                    )
                    q_vals_hats.append(q_val_sims)

    return keys, q_vals_hats


def do_one_sample(scorer, c_pucts, n_sents, prior_choice):
    s = scorer.scores.mean(-1)[:n_sents, :n_sents, :n_sents]
    if prior_choice == "best":
        selected_sents = np.array(np.unravel_index(s.argmax(), s.shape))
    elif prior_choice == "worst":
        s_pos = np.ma.masked_less_equal(s, 0)
        selected_sents = np.array(np.unravel_index(s_pos.argmin(), s_pos.shape))
    else:
        # Get median
        selected_sents = np.array([a[0] for a in np.nonzero(s == np.percentile(s, 50, interpolation="nearest"))])

    grdy = np.ones((3,)) / 3
    greedy = np.zeros((n_sents,))
    greedy[selected_sents] = grdy

    return [
        (
            c_puct,
            prior_choice,
            [collect_sim(scorer, c_puct, n_sents, greedy, tau) for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]],
        )
        for c_puct in c_pucts
    ]


def collect_sim(scorer, c_puct, n_sents, greedy, tau, n_samples=250):
    unif = np.ones((n_sents,)) / n_sents

    priors = (1 - tau) * unif + tau * greedy
    priors[priors < 0] = 0
    priors /= priors.sum()

    n_sents = min(scorer.scores.shape[0], 50)

    n_visits = np.zeros(n_sents, dtype=int)
    q_vals = np.zeros(n_sents, dtype=np.float32)
    q_vals_sims = np.zeros(n_samples, dtype=np.float32)

    if np.isnan(priors).any():
        return None, None, None

    best_idxs = np.argpartition(priors, -3)[-3:]
    prior_max_score = scorer.scores[tuple(best_idxs)].mean()
    prior_max_proba = priors[best_idxs].sum()

    for n in range(1, n_samples + 1):
        ucb = q_vals + c_puct * priors * np.sqrt(2 * np.log(n) / n_visits)
        ucb = np.nan_to_num(ucb, nan=np.inf)
        threshold = np.partition(ucb, -3)[-3]
        elligible_idxs = np.argwhere(ucb >= threshold)[:, 0]
        sampled_idxs = np.random.choice(elligible_idxs, 3, replace=False)
        summ_score = scorer.scores[tuple(sampled_idxs)].mean()
        q_vals[sampled_idxs] = (summ_score + q_vals[sampled_idxs] * n_visits[sampled_idxs]) / (
            n_visits[sampled_idxs] + 1
        )
        n_visits[sampled_idxs] += 1

        threshold = np.partition(q_vals, -3)[-3]
        elligible_idxs = np.argwhere(q_vals >= threshold)[:, 0]
        best_idxs = np.random.choice(elligible_idxs, 3, replace=False)
        q_vals_sims[n - 1] = scorer.scores[tuple(best_idxs)].mean()

    return q_vals_sims, prior_max_score, prior_max_proba, tau
