import pickle

from src.domain.loader_utils import TextDataCollator
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from itertools import combinations
from collections import defaultdict, namedtuple
from scipy.special import entr
from joblib import Parallel, delayed


class BanditSumMCSExperiment(pl.LightningModule):
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

        self.mcs_log_path = "/project/def-adurand/magod/rl_summ/mcs_exp"
        os.makedirs(self.mcs_log_path, exist_ok=True)
        self.mcs_log_path += "/results.pck"
        with open(self.mcs_log_path, "wb") as f:
            pickle.dump({}, f)

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

    def __get_reward_scorers(self, ids, subset):
        if subset in ["train", "val", "test"]:
            return [self.reward_builder.init_scorer(id, subset) for id in ids]
        else:
            raise ValueError(
                f'Bad subset : {subset}. Should be one of ["train", "val", "test].'
            )

    def mcs_exp(self, scorers, ids):
        return Parallel(n_jobs=-1, verbose=1, backend="loky")(
            collect_sims(scorer, id) for scorer, id in zip(scorers, ids)
        )

    def forward(self, batch, subset):
        raw_contents, contents, raw_abstracts, abstracts, ids = batch
        torch.set_grad_enabled(False)

        scorers = self.__get_reward_scorers(ids, subset)

        results = self.mcs_exp(scorers, ids)
        keys = [key for r in results for key in r[0]]
        f_hats = [vals for r in results for vals in r[1]]

        return (keys, f_hats)

    def get_step_output(self, loss, greedy_rewards, mcts_rewards, max_scores):
        output_dict = {}

        log_dict = {
            "greedy_rouge_mean": greedy_rewards.mean(),
            "mcts_rouge_mean": mcts_rewards.mean(),
            "max_rouge_mean": max_scores.mean(),
        }
        log_dict["loss"] = loss

        output_dict["log"] = log_dict

        if "loss" in log_dict:
            output_dict["loss"] = log_dict["loss"]

        tqdm_keys = ["mcts_rouge", "greedy_rouge", "max_rouge"]
        output_dict["progress_bar"] = {k: log_dict[f"{k}_mean"] for k in tqdm_keys}

        return output_dict

    def training_step(self, batch, batch_idx):
        greedy_rewards, loss, mcts_rewards, max_scores = self.forward(
            batch, subset="train"
        )

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
        return None

    def test_step(self, batch, batch_idx):
        keys, f_hats = self.forward(batch, subset="test")

        with open(self.mcs_log_path, "rb") as f:
            d = pickle.load(f)

        for key, val in zip(keys, f_hats):
            d[key] = val

        with open(self.mcs_log_path, "wb") as f:
            pickle.dump(d, f)

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
                k: v
                for k, v in log_dict.items()
                if any([t_k in k for t_k in tqdm_keys])
            }

        return combined_outputs

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

        self.lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.1, verbose=True
        )

        return optimizer

    def train_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(
                self.fields, self.reward_builder, subset="train"
            ),
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(
                self.fields, self.reward_builder, subset="train"
            ),
            batch_size=self.test_batch_size,
            drop_last=False,
        )

    def test_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(
                self.fields, self.reward_builder, subset="train"
            ),
            batch_size=self.test_batch_size,
            drop_last=False,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return BanditSumMCSExperiment(
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


@delayed
def collect_sims(scorer, id):
    keys = []
    f_hats = []

    n_sents = min(scorer.scores.shape[0], 50)
    combs = list(combinations(range(n_sents), 3))
    results = [
        collect_sim(scorer, tau, combs, n_sents)
        for tau in np.linspace(0.0, 1.0, num=101)
    ]

    for f, entropy, top3, f_sims in results:
        if f:
            keys.append((f, entropy, top3, id, n_sents))
            f_hats.append(f_sims)

    return keys, f_hats


def collect_sim(scorer, tau, combs, n_sents, n_samples=1000):
    unif = np.ones((n_sents,)) / n_sents
    selected_sents = np.array(combs[np.random.choice(len(combs))])
    noise = np.random.normal(scale=0.1, size=(3,))
    noise -= noise.mean()
    grdy = np.ones((3,)) / 3
    grdy += noise
    greedy = np.zeros((n_sents,))
    greedy[selected_sents] = grdy

    distro = (1 - tau) * unif + tau * greedy
    distro[distro < 0] = 0
    distro /= distro.sum()

    comb_probas = np.array([distro[i] * distro[j] * distro[k] for i, j, k in combs])
    comb_probas /= comb_probas.sum()

    if np.isnan(distro).any() or np.isnan(comb_probas).any():
        return None, None, None, None

    f = np.sum(
        [proba * scorer(i, j, k) for (i, j, k), proba in zip(combs, comb_probas)]
    )
    ent = entr(comb_probas).sum()
    top3 = np.partition(distro, -3)[-3:].sum()

    sampled_combs = np.random.choice(
        len(combs), size=n_samples, replace=True, p=comb_probas
    )
    f_sims = [scorer(combs[comb]) for comb in sampled_combs]
    f_sims = np.cumsum(f_sims) / np.arange(start=1, stop=n_samples + 1)

    return f, ent, top3, f_sims
