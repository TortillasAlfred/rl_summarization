from src.domain.loader_utils import TextDataCollator, NGRAMSLoader
from src.domain.linucb import LinUCBProcess

import time
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
import os


class LinSITModel(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super().__init__()
        self.hparams = hparams
        self.fields = dataset.fields
        self.pad_idx = dataset.pad_idx
        self.reward_builder = reward
        self.data_path = hparams.data_path

        self.embedding_dim = dataset.embedding_dim
        self.pad_idx = dataset.pad_idx
        self.splits = dataset.get_splits()
        self.n_epochs_done = 0

        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size
        self.hidden_dim = hparams.hidden_dim
        self.decoder_dim = hparams.decoder_dim
        self.learning_rate = hparams.learning_rate
        self.n_sents_per_summary = hparams.n_sents_per_summary
        self.c_puct = hparams.c_puct
        self.ucb_sampling = hparams.ucb_sampling
        self.weight_decay = hparams.weight_decay
        self.batch_idx = 0

        self.__build_model(dataset)
        self.model = RLSummModel(hparams.hidden_dim, hparams.decoder_dim,)

        if hparams.n_jobs_for_mcts == -1:
            self.n_processes = os.cpu_count()
        else:
            self.n_processes = hparams.n_jobs_for_mcts
        self.pool = mp.Pool(processes=self.n_processes)

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

        return affinities, valid_sentences

    def forward(self, batch, subset):
        if subset == "train":
            (
                raw_contents,
                contents,
                raw_abstracts,
                abstracts,
                ids,
                scorers,
                n_grams_dense,
            ) = batch
        else:
            (raw_contents, contents, raw_abstracts, abstracts, ids, scorers,) = batch
        batch_size = len(contents)

        self.wl_encoder.flatten_parameters()
        self.model.sl_encoder.flatten_parameters()

        action_vals, valid_sentences = self.__extract_features(contents)
        _, greedy_idxs = torch.topk(action_vals, self.n_sents_per_summary, sorted=False)

        if subset == "train":
            greedy_rewards = []
            for scorer, sent_idxs in zip(scorers, greedy_idxs):
                greedy_rewards.append(scorer(sent_idxs.tolist()))
            greedy_rewards = torch.tensor(greedy_rewards)

            linucb_results = self.pool.map(
                LinUCBProcess(self.ucb_sampling, self.c_puct),
                zip(scorers, n_grams_dense),
            )

            linucb_targets = torch.stack([r[0] for r in linucb_results]).to(
                action_vals.device
            )
            linucb_deltas = torch.tensor([r[1] for r in linucb_results])

            loss = (linucb_targets - action_vals) ** 2 * valid_sentences
            loss = loss.sum(-1) / valid_sentences.sum(-1)
            loss = loss.sum()

            return greedy_rewards, loss, linucb_deltas
        else:
            greedy_rewards = scorers.get_scores(
                greedy_idxs, raw_contents, raw_abstracts
            )

            return torch.from_numpy(greedy_rewards)

    def training_step(self, batch, batch_idx):
        start = time.time()
        greedy_rewards, loss, ucb_deltas = self.forward(batch, subset="train")
        end = time.time()

        log_dict = {
            "greedy_rouge_mean": greedy_rewards.mean(),
            "ucb_deltas": ucb_deltas.mean(),
            "loss": loss.detach(),
            "batch_time": end - start,
        }

        for key, val in log_dict.items():
            self.log(key, val, prog_bar="greedy" in key)

        return loss

    def validation_step(self, batch, batch_idx):
        greedy_rewards = self.forward(batch, subset="val")

        reward_dict = {
            "val_greedy_rouge_1": greedy_rewards[:, 0],
            "val_greedy_rouge_2": greedy_rewards[:, 1],
            "val_greedy_rouge_L": greedy_rewards[:, 2],
            "val_greedy_rouge_mean": greedy_rewards.mean(-1),
        }

        for name, val in reward_dict.items():
            self.log(name, val, prog_bar="mean" in name)

        return reward_dict["val_greedy_rouge_mean"]

    def validation_epoch_end(self, outputs):
        mean_rouge = torch.stack(outputs).mean()
        self.lr_scheduler.step(mean_rouge)

        current_lr = self.trainer.optimizers[0].param_groups[1]["lr"]
        self.log("lr", current_lr)

    def test_step(self, batch, batch_idx):
        greedy_rewards = self.forward(batch, subset="test")

        reward_dict = {
            "test_greedy_rouge_1": greedy_rewards[:, 0],
            "test_greedy_rouge_2": greedy_rewards[:, 1],
            "test_greedy_rouge_L": greedy_rewards[:, 2],
            "test_greedy_rouge_mean": greedy_rewards.mean(-1),
        }

        for name, val in reward_dict.items():
            self.log(name, val)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.embeddings.parameters(),
                    "lr": self.learning_rate * 0.1,
                },
                {"params": self.wl_encoder.parameters()},
                {"params": self.model.sl_encoder.parameters()},
                {"params": self.model.decoder.parameters()},
            ],
            lr=self.learning_rate,
            betas=[0, 0.999],
            weight_decay=self.weight_decay,
        )

        self.lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.2, verbose=True
        )

        return optimizer

    def train_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(
                self.fields,
                self.reward_builder,
                subset="train",
                n_grams_loader=NGRAMSLoader(self.data_path),
            ),
            batch_size=self.train_batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        dataset = self.splits["val"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.fields, self.reward_builder, subset="val"),
            batch_size=self.test_batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        dataset = self.splits["test"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(
                self.fields, self.reward_builder, subset="test"
            ),
            batch_size=self.test_batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return LinSITModel(dataset, reward, config,)


class RLSummModel(torch.nn.Module):
    def __init__(
        self, hidden_dim, decoder_dim,
    ):
        super().__init__()
        self.sl_encoder = torch.nn.LSTM(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, decoder_dim),
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

