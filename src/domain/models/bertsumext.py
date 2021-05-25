from src.domain.loader_utils import TextDataCollator
from src.domain.ucb import UCBProcess

import time
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
import os

"""
Informations Pertinentes sur sit.py:

In: dataset
Out: <src.domain.dataset.CnnDailyMailDataset object at 0x7f666ac4afd0>

In: dataset.fields
Out: [('raw_content', <torchtext.data.fiel...66ac4af70>), ('content', <torchtext.data.fiel...66ac38c70>), ('raw_abstract', <torchtext.data.fiel...66ac38f10>), ('abstract', <torchtext.data.fiel...66ac38eb0>), ('id', <torchtext.data.fiel...66ac38df0>)]
special variables
function variables
0:('raw_content', <torchtext.data.fiel...66ac4af70>)
1:('content', <torchtext.data.fiel...66ac38c70>)
2:('raw_abstract', <torchtext.data.fiel...66ac38f10>)
3:('abstract', <torchtext.data.fiel...66ac38eb0>)
4:('id', <torchtext.data.fiel...66ac38df0>)
len():5

In: dataset.get_splits()
Out: {'train': <src.domain.dataset....653f955b0>, 'val': <src.domain.dataset....6533ee160>, 'test': <src.domain.dataset....6533ee190>}

In: dataset.get_splits()["train"]
Out: <src.domain.dataset.TextDataset object at 0x7f6653f953d0>

In: dataset.embedding_dim
Out: 100

In: dataset.pad_idx
Out: 1

"""


class BertSumExt(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super().__init__()
        self.hparams = hparams
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
        self.learning_rate = hparams.learning_rate
        self.n_sents_per_summary = hparams.n_sents_per_summary
        self.c_puct = hparams.c_puct
        self.ucb_sampling = hparams.ucb_sampling
        self.weight_decay = hparams.weight_decay
        self.batch_idx = 0
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.tensor_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.idxs_repart = torch.zeros(
            50, dtype=torch.float32, device=self.tensor_device
        )
        self.test_size = len(self.splits["test"])
        self.targets_repart = torch.zeros(
            50, dtype=torch.float64, device=self.tensor_device
        )
        self.train_size = len(self.splits["train"])

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
        affinities = affinities + valid_sentences.float().log()

        return affinities, valid_sentences

    def forward(self, batch, subset):
        raw_contents, contents, raw_abstracts, abstracts, ids, scorers = batch
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

            ucb_results = self.pool.map(
                UCBProcess(self.ucb_sampling, self.c_puct), scorers,
            )

            ucb_targets = torch.tensor(
                [r[0] for r in ucb_results], device=action_vals.device
            )
            ucb_deltas = torch.tensor([r[1] for r in ucb_results])

            # Softmax
            target_distro = 10 ** (-10 * (1 - ucb_targets)) * valid_sentences

            loss = self.criterion(action_vals, target_distro)
            loss[~valid_sentences] = 0.0
            loss = loss.sum(-1) / valid_sentences.sum(-1)
            loss = loss.mean()

            self.targets_repart += target_distro.sum(0)

            return greedy_rewards, loss, ucb_deltas
        else:
            greedy_rewards = scorers.get_scores(
                greedy_idxs, raw_contents, raw_abstracts
            )

            if subset == "test":
                idxs_repart = torch.zeros_like(action_vals)
                idxs_repart.scatter_(1, greedy_idxs, 1)

                self.idxs_repart += idxs_repart.sum(0)

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

    def training_epoch_end(self, outputs):
        for i, idx_repart in enumerate(self.targets_repart / self.train_size):
            self.log(f"targets_idx_{i}", idx_repart)

        self.targets_repart.zero_()

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

        return reward_dict["val_greedy_rouge_mean"].mean()

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

    def test_epoch_end(self, outputs):
        for i, idx_repart in enumerate(self.idxs_repart / self.test_size):
            self.log(f"idx_{i}", idx_repart)

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
            optimizer, mode="max", patience=5, factor=0.2, verbose=True
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
            num_workers=self.num_workers,
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
            num_workers=self.num_workers,
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
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return BertSumExt(dataset, reward, config,)


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
        )

    def sentence_level_encoding(self, contents):
        sent_contents, _ = self.sl_encoder(contents)

        return sent_contents

    def produce_affinities(self, sent_contents):
        affinities = self.decoder(sent_contents).squeeze(-1)

        return affinities

    def forward(self, x):
        pass

