from src.domain.rewards.rouge_python import RougePythonReward
import src.domain.mcts_oful_exp as mcts_oful
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


class RLSumOFULEXP(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super().__init__()
        self.hparams = hparams
        self.dataset = dataset
        self.reward_builder = reward

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
        self.raw_run_done = False

        self.mcts_log_path = os.path.join(
            "/project/def-lulam50/magod/rl_summ/mcts_exp", f"alpha_{self.alpha_oful}"
        )
        os.makedirs(self.mcts_log_path, exist_ok=True)

        self.raw_path = os.path.join(self.mcts_log_path, "raw")
        self.pretrained_path = os.path.join(self.mcts_log_path, "pretrained")

        if hparams.raw_run == 0:
            os.makedirs(self.raw_path, exist_ok=True)
            with open(os.path.join(self.raw_path, "results.pck"), "wb") as f:
                pickle.dump(
                    {
                        "n_sents": [],
                        "max_scores": [],
                        "theta_hat_predictions": [],
                        "regrets": [],
                        "times": [],
                    },
                    f,
                )
        else:
            os.makedirs(self.pretrained_path, exist_ok=True)
            with open(os.path.join(self.pretrained_path, "results.pck"), "wb") as f:
                pickle.dump(
                    {
                        "n_sents": [],
                        "max_scores": [],
                        "theta_hat_predictions": [],
                        "regrets": [],
                        "times": [],
                    },
                    f,
                )

        self.lock = threading.Lock()

        mp.set_start_method("forkserver", force=True)
        if hparams.n_jobs_for_mcts == -1:
            self.n_processes = os.cpu_count()
        else:
            self.n_processes = hparams.n_jobs_for_mcts
        self.pools = [
            mp.Pool(processes=self.n_processes)
            for _ in range(torch.cuda.device_count())
        ]

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

    def warmup_oful(self, valid_sentences, scorers):
        all_sampled_summs = []
        all_scores = []

        for valid_sents, scorer in zip(valid_sentences, scorers):
            all_sums = list(combinations(list(range(valid_sents.sum())), 3))
            sampled_summs = np.random.choice(len(all_sums), 250, replace=True)
            sampled_summs = [all_sums[summ] for summ in sampled_summs]
            scores = torch.tensor(
                [scorer.scores[tuple(summ)] for summ in sampled_summs],
                device=self.device,
            )
            all_scores.append(scores)
            all_sampled_summs.append(sampled_summs)

        all_scores = torch.cat(all_scores)

        return all_sampled_summs, all_scores

    def mcts_oful(
        self, sent_contents, doc_contents, valid_sentences, scorers, gpu_device_idx
    ):
        mcts_pures = self.pools[gpu_device_idx].map(
            mcts_oful.RLSumOFULValueProcess(
                n_samples=self.n_mcts_samples,
                lambda_oful=self.lambda_oful,
                alpha_oful=self.alpha_oful,
                action_dim=sent_contents.shape[-1],
                device=sent_contents.device,
                n_sents_per_summary=self.n_sents_per_summary,
            ),
            zip(
                sent_contents,
                doc_contents,
                valid_sentences,
                [s.scores for s in scorers],
            ),
        )

        mcts_theta_hats = torch.stack([m[0] for m in mcts_pures]).transpose(2, 1)
        max_scores = torch.stack([m[1] for m in mcts_pures])
        n_sents = torch.stack([m[2] for m in mcts_pures])
        theta_hat_predictions = torch.stack([m[3] for m in mcts_pures]).T
        regrets = torch.stack([m[4] for m in mcts_pures]).T
        times = torch.stack([m[5] for m in mcts_pures])

        return (
            mcts_theta_hats,
            max_scores,
            n_sents,
            theta_hat_predictions,
            regrets,
            times,
        )

    def __get_reward_scorers(self, ids, subset, gpu_idx, batch_size):
        if subset in ["train", "val", "test"]:
            return [
                self.reward_builder.init_scorer(id, subset)
                for id in ids[gpu_idx * batch_size : (gpu_idx + 1) * batch_size]
            ]
        # elif subset in ["val", "test"]:
        #     return [RougePythonReward() for _ in range(batch_size)]
        else:
            raise ValueError(
                f'Bad subset : {subset}. Should be one of ["train", "val", "test].'
            )

    def forward(self, batch, subset):
        raw_contents, contents, raw_abstracts, abstracts, ids = batch
        gpu_idx = contents.device.index
        batch_size = len(contents)
        torch.set_grad_enabled(False)

        self.wl_encoder.flatten_parameters()
        self.model.sl_encoder.flatten_parameters()

        scorers = self.__get_reward_scorers(ids, subset, gpu_idx, batch_size)

        (
            sent_contents,
            doc_contents,
            valid_sentences,
            raw_sent_contents,
        ) = self.__extract_features(contents)

        if subset == "train":
            if self.batch_idx >= self.warmup_batches:
                mcts_theta_hats, max_scores = self.mcts_oful(
                    sent_contents, doc_contents, valid_sentences, scorers, gpu_idx,
                )

                mcts_rewards = []

                for theta_hat_doc, sent_conts, val_sents, scorer in zip(
                    mcts_theta_hats, sent_contents, valid_sentences, scorers
                ):
                    sent_predicted_vals = theta_hat_doc.mm(sent_conts.T)
                    sent_predicted_vals = sent_predicted_vals.squeeze()[val_sents]
                    _, selected_sents = sent_predicted_vals.topk(
                        self.n_sents_per_summary
                    )
                    mcts_rewards.append(
                        torch.from_numpy(scorer.get_score(selected_sents.tolist()))
                    )

                mcts_rewards = torch.stack(mcts_rewards)

                torch.set_grad_enabled(True)

                (
                    sent_contents,
                    doc_contents,
                    valid_sentences,
                    _,
                ) = self.__extract_features(contents)
                theta_hats = self.model.produce_theta_hat(doc_contents)

                greedy_rewards = []

                for theta_hat_doc, sent_conts, val_sents, scorer in zip(
                    theta_hats, sent_contents, valid_sentences, scorers
                ):
                    sent_predicted_vals = theta_hat_doc.mm(sent_conts.T)
                    sent_predicted_vals = sent_predicted_vals.squeeze()[val_sents]
                    _, selected_sents = sent_predicted_vals.topk(
                        self.n_sents_per_summary
                    )
                    greedy_rewards.append(
                        torch.from_numpy(scorer.get_score(selected_sents.tolist()))
                    )

                greedy_rewards = torch.stack(greedy_rewards)

                loss = (mcts_theta_hats.to(valid_sentences.device) - theta_hats) ** 2
                loss = loss.sum()

                return greedy_rewards, loss, mcts_rewards, max_scores
            else:
                sampled_summs, sampled_scores = self.warmup_oful(
                    valid_sentences, scorers
                )

                theta_hats = self.model.produce_theta_hat(doc_contents)

                greedy_rewards = []

                for theta_hat_doc, sent_conts, val_sents, scorer in zip(
                    theta_hats, sent_contents, valid_sentences, scorers
                ):
                    sent_predicted_vals = theta_hat_doc.mm(sent_conts.T)
                    sent_predicted_vals = sent_predicted_vals.squeeze()[val_sents]
                    _, selected_sents = sent_predicted_vals.topk(
                        self.n_sents_per_summary
                    )
                    greedy_rewards.append(
                        torch.from_numpy(scorer.get_score(selected_sents.tolist()))
                    )

                greedy_rewards = torch.stack(greedy_rewards)

                torch.set_grad_enabled(True)

                (
                    sent_contents,
                    doc_contents,
                    valid_sentences,
                    _,
                ) = self.__extract_features(contents)
                predicted_scores = self.model.pretraining_output(
                    sent_contents, doc_contents, sampled_summs
                )

                loss = (
                    sampled_scores.to(valid_sentences.device) - predicted_scores
                ) ** 2
                loss = loss.sum()

                return (
                    greedy_rewards,
                    loss,
                    torch.zeros_like(greedy_rewards),
                    torch.zeros_like(greedy_rewards),
                )
        elif subset == "test":
            (
                mcts_theta_hats,
                max_scores,
                n_sents,
                theta_hat_predictions,
                regrets,
                times,
            ) = self.mcts_oful(
                sent_contents, doc_contents, valid_sentences, scorers, gpu_idx,
            )

            theta_hats = self.model.produce_theta_hat(doc_contents)
            greedy_rewards = []
            decal = gpu_idx * batch_size

            for (i, (theta_hat_doc, sent_conts, val_sents, scorer,),) in enumerate(
                zip(theta_hats, sent_contents, valid_sentences, scorers,)
            ):
                sent_predicted_vals = theta_hat_doc.mm(sent_conts.T)
                sent_predicted_vals = sent_predicted_vals.squeeze()[val_sents]
                _, selected_sents = sent_predicted_vals.topk(self.n_sents_per_summary)
                greedy_rewards.append(
                    torch.from_numpy(scorer.get_score(selected_sents.tolist()))
                )

            greedy_rewards = torch.stack(greedy_rewards)

            return (
                greedy_rewards.to(self.device),
                n_sents.to("cpu"),
                max_scores.to("cpu"),
                theta_hat_predictions.to("cpu"),
                regrets.to("cpu"),
                times.to("cpu"),
            )
        else:
            theta_hats = self.model.produce_theta_hat(doc_contents)
            greedy_rewards = []
            decal = gpu_idx * batch_size

            for (i, (theta_hat_doc, sent_conts, val_sents, scorer,),) in enumerate(
                zip(theta_hats, sent_contents, valid_sentences, scorers,)
            ):
                sent_predicted_vals = theta_hat_doc.mm(sent_conts.T)
                sent_predicted_vals = sent_predicted_vals.squeeze()[val_sents]
                _, selected_sents = sent_predicted_vals.topk(self.n_sents_per_summary)
                greedy_rewards.append(
                    torch.from_numpy(scorer.get_score(selected_sents.tolist()))
                )

            greedy_rewards = torch.stack(greedy_rewards)

            return greedy_rewards.to(self.device)

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

        output_dict["log"]["learning_rate"] = self.trainer.optimizers[0].param_groups[
            1
        ]["lr"]

        return output_dict

    def test_step(self, batch, batch_idx):
        (
            greedy_rewards,
            n_sents,
            max_scores,
            theta_hat_predictions,
            regrets,
            times,
        ) = self.forward(batch, subset="test")

        reward_dict = {
            "test_greedy_rouge_1": greedy_rewards[:, 0],
            "test_greedy_rouge_2": greedy_rewards[:, 1],
            "test_greedy_rouge_L": greedy_rewards[:, 2],
            "test_greedy_rouge_mean": greedy_rewards.mean(-1),
        }

        if self.raw_run_done:
            pickle_path = os.path.join(self.pretrained_path, "results.pck")
        else:
            pickle_path = os.path.join(self.raw_path, "results.pck")

        self.lock.acquire()
        try:
            with open(pickle_path, "rb") as f:
                d = pickle.load(f)

            d["n_sents"].append(n_sents)
            d["max_scores"].append(max_scores)
            d["theta_hat_predictions"].append(theta_hat_predictions)
            d["regrets"].append(regrets)
            d["times"].append(times)

            with open(pickle_path, "wb") as f:
                pickle.dump(d, f)
        finally:
            self.lock.release()

        return reward_dict

    def test_step_end(self, outputs):
        for vals in outputs.values():
            vals = vals.mean()

        return outputs

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
            collate_fn=text_data_collator(dataset),
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        dataset = self.splits["val"]
        return DataLoader(
            dataset,
            collate_fn=text_data_collator(dataset),
            batch_size=self.test_batch_size,
            drop_last=True,
        )

    def test_dataloader(self):
        dataset = self.splits["val"]
        return DataLoader(
            dataset,
            collate_fn=text_data_collator(dataset),
            batch_size=self.test_batch_size,
            drop_last=True,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return RLSumOFULEXP(dataset, reward, config,)


def text_data_collator(dataset):
    def collate(data):
        batch = defaultdict(list)

        for datum in data:
            for name, field in dataset.fields.items():
                batch[name].append(field.preprocess(getattr(datum, name)))

        batch = {
            name: field.process(batch[name]) for name, field in dataset.fields.items()
        }

        batch = namedtuple("batch", batch.keys())(**batch)

        return batch

    return collate


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
