from src.domain.loader_utils import TextDataCollator
from src.domain.linsit import LinSITExpProcess

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import combinations
import random
import os
import torch.multiprocessing as mp
import pickle


class LinSITExp(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super(LinSITExp, self).__init__()
        self.fields = dataset.fields
        self.reward_builder = reward

        self.embedding_dim = dataset.embedding_dim
        self.pad_idx = dataset.pad_idx
        self.splits = dataset.get_splits()
        self.n_epochs_done = 0

        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size
        self.hidden_dim = hparams.hidden_dim
        self.decoder_dim = hparams.decoder_dim
        self.learning_rate = hparams.learning_rate
        self.epsilon = hparams.epsilon
        self.n_sents_per_summary = hparams.n_sents_per_summary
        self.dropout = hparams.dropout
        self.weight_decay = hparams.weight_decay
        self.pretraining_path = hparams.pretraining_path
        self.log_path = hparams.log_path
        self.n_mcts_samples = hparams.n_mcts_samples
        self.batch_idx = 0

        os.makedirs(self.log_path, exist_ok=True)
        self.__build_model(dataset)
        self.model = RLSummModel(hparams.hidden_dim, hparams.decoder_dim, self.dropout,)

        mp.set_start_method("forkserver", force=True)
        mp.set_sharing_strategy("file_system")

        if hparams.n_jobs_for_mcts == -1:
            self.n_processes = os.cpu_count()
        else:
            self.n_processes = hparams.n_jobs_for_mcts
        self.pools = [
            mp.Pool(processes=self.n_processes)
            for _ in range(torch.cuda.device_count())
        ]

    def __build_model(self, dataset):
        self.embeddings = torch.nn.Embedding.from_pretrained(
            dataset.vocab.vectors, freeze=False, padding_idx=self.pad_idx
        )
        self.wl_encoder = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
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
        sent_contents = self.model.sentence_level_encoding(contents)
        affinities = self.model.produce_affinities(sent_contents)
        affinities = affinities * valid_sentences

        return affinities, valid_sentences, sent_contents

    def linsit_exp(
        self,
        sent_contents,
        valid_sentences,
        priors,
        scorers,
        ids,
        c_pucts,
        n_pretraining_steps,
        gpu_device_idx,
    ):
        results = self.pools[gpu_device_idx].map(
            LinSITExpProcess(
                n_samples=self.n_mcts_samples,
                c_pucts=c_pucts,
                n_pretraining_steps=n_pretraining_steps,
                device=sent_contents.device,
            ),
            zip(
                sent_contents, valid_sentences, priors, [s.scores for s in scorers], ids
            ),
        )

        return [r for res in results for r in res]

    def get_device_nontensors(
        self, raw_contents, raw_abstracts, ids, scorers, gpu_idx, batch_size
    ):
        begin_idx = gpu_idx * batch_size
        end_idx = (gpu_idx + 1) * batch_size

        return (
            raw_contents[begin_idx:end_idx],
            raw_abstracts[begin_idx:end_idx],
            ids[begin_idx:end_idx],
            scorers[begin_idx:end_idx],
        )

    def forward(self, batch, subset):
        raw_contents, contents, raw_abstracts, abstracts, ids, scorers = batch.values()
        batch_size = len(contents)

        gpu_idx = contents.device.index
        raw_contents, raw_abstracts, ids, scorers = self.get_device_nontensors(
            raw_contents, raw_abstracts, ids, scorers, gpu_idx, batch_size
        )

        torch.set_grad_enabled(False)

        self.wl_encoder.flatten_parameters()
        self.model.sl_encoder.flatten_parameters()

        c_pucts = np.logspace(-1, 5, 7)

        (_, valid_sentences, sent_contents) = self.__extract_features(contents)
        priors = torch.ones_like(valid_sentences, dtype=torch.float32)
        priors /= valid_sentences.sum(-1, keepdim=True)

        all_keys = []
        all_theta_hat_predictions = []

        for n_pretraining_steps in [1, 100, 1000]:
            self.load_from_n_pretraining_steps(n_pretraining_steps)

            (_, valid_sentences, sent_contents) = self.__extract_features(contents)

            results = self.linsit_exp(
                sent_contents,
                valid_sentences,
                priors,
                scorers,
                ids,
                c_pucts,
                n_pretraining_steps,
                gpu_idx,
            )

            keys = [r[0] for r in results]
            theta_hat_predictions = [r[1] for r in results]

            all_keys.extend(keys)
            all_theta_hat_predictions.extend(
                [t.cpu().numpy() for t in theta_hat_predictions]
            )

        return all_keys, all_theta_hat_predictions, gpu_idx

    def load_from_n_pretraining_steps(self, n_steps):
        load_path = f"{self.pretraining_path}/{n_steps}_batches.pt"

        state_dict = torch.load(load_path)
        state_dict = {k: v for k, v in state_dict.items() if "embeddings" not in k}

        self.load_state_dict(state_dict, strict=False)

    def get_step_output(self, loss, greedy_rewards, generated_rewards):
        output_dict = {}

        log_dict = {
            "greedy_rouge_1": greedy_rewards[:, 0].mean(),
            "greedy_rouge_2": greedy_rewards[:, 1].mean(),
            "greedy_rouge_L": greedy_rewards[:, 2].mean(),
            "greedy_rouge_mean": greedy_rewards.mean(-1).mean(),
            "generated_rouge_1": generated_rewards[:, 0].mean(),
            "generated_rouge_2": generated_rewards[:, 1].mean(),
            "generated_rouge_L": generated_rewards[:, 2].mean(),
            "generated_rouge_mean": generated_rewards.mean(-1).mean(),
        }
        log_dict["loss"] = loss

        output_dict["log"] = log_dict

        if "loss" in log_dict:
            output_dict["loss"] = log_dict["loss"]

        tqdm_keys = ["greedy_rouge", "generated_rouge"]
        output_dict["progress_bar"] = {k: log_dict[f"{k}_mean"] for k in tqdm_keys}

        return output_dict

    def training_step(self, batch, batch_idx):
        generated_rewards, loss, greedy_rewards = self.forward(batch, subset="train")

        return self.get_step_output(
            loss=loss.to(self.device),
            greedy_rewards=greedy_rewards.to(self.device),
            generated_rewards=generated_rewards.to(self.device),
        )

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

        self.lr_scheduler.step(output_dict["log"]["val_greedy_rouge_mean"])

        output_dict["log"]["learning_rate"] = self.trainer.optimizers[0].param_groups[
            1
        ]["lr"]

        return output_dict

    def test_step(self, batch, batch_idx):
        keys, theta_hat_predictions, gpu_idx = self.forward(batch, subset="test")

        d = {}

        for key, preds in zip(keys, theta_hat_predictions):
            d[key] = preds

        with open(
            os.path.join(self.log_path, f"results_{batch_idx}_{gpu_idx}.pck"), "wb"
        ) as f:
            pickle.dump(d, f)

    def test_step_end(self, outputs):
        pass

    def generic_epoch_end(self, outputs, is_test=False):
        combined_outputs = {}
        log_dict = {}

        for key in outputs[0]:
            log_dict[key] = torch.hstack([output[key] for output in outputs]).mean()

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
        all_dict_paths = os.listdir(self.log_path)
        d = {}

        for path in all_dict_paths:
            with open(os.path.join(self.log_path, path), "rb") as f:
                d_i = pickle.load(f)

            for k, v in d_i.items():
                d[k] = v

        with open(os.path.join(self.log_path, "results.pck"), "wb") as f:
            pickle.dump(d, f)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.embeddings.parameters(),
                    "lr": self.learning_rate * 0.1,
                },
                {"params": self.wl_encoder.parameters()},
                {"params": self.model.sl_encoder.parameters()},
                {
                    "params": self.model.decoder.parameters(),
                    "lr": self.learning_rate * 0.1,
                },
                {
                    "params": self.model.pretraining_decoder.parameters(),
                    "lr": self.learning_rate * 0.1,
                },
            ],
            lr=self.learning_rate,
            betas=[0, 0.999],
            weight_decay=self.weight_decay,
        )

        self.lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.1, verbose=True
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
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(
                self.fields, self.reward_builder, subset="train"
            ),
            batch_size=self.test_batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(
                self.fields, self.reward_builder, subset="train"
            ),
            batch_size=self.test_batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return LinSITExp(dataset, reward, config,)


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
        self.pretraining_decoder = torch.nn.Linear(hidden_dim * 2, 1, bias=False)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, decoder_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, 1),
            torch.nn.Sigmoid(),
        )

    def sentence_level_encoding(self, contents):
        sent_contents, _ = self.sl_encoder(contents)

        return sent_contents

    def produce_affinities(self, sent_contents):
        affinities = self.decoder(sent_contents).squeeze(-1)

        return affinities

    def forward(self, x):
        pass
