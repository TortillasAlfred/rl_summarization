import os
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.domain.ucb import BertUCBProcess
from src.domain.loader_utils import TextDataCollator
from ..bertsum_transformer import Summarizer

# This is a version of bertbanditsum.py which has been modified to predict
# the three first sentences of the article (Lead-3)


class BertBinarySumLead3(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super().__init__()
        self.tensor_device = "cuda" if hparams.gpus > 0 and torch.cuda.is_available() else "cpu"

        self.hparams = hparams
        self.colname_2_field_objs = dataset.fields

        self.pad_idx = dataset.pad_idx
        self.splits = dataset.get_splits()
        self.reward_builder = reward
        self.n_epochs_done = 0
        self.first_val_batches_done = False

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
        self.predicted_idxs_repart = {
            "train": torch.zeros(50, dtype=torch.float32, device=self.tensor_device),
            "val": torch.zeros(50, dtype=torch.float32, device=self.tensor_device),
            "test": torch.zeros(50, dtype=torch.float32, device=self.tensor_device),
        }
        self.target_idxs_repart = {
            "train": torch.zeros(50, dtype=torch.float32, device=self.tensor_device),
            "val": torch.zeros(50, dtype=torch.float32, device=self.tensor_device),
            "test": torch.zeros(50, dtype=torch.float32, device=self.tensor_device),
        }
        self.test_size = len(self.splits["test"])
        self.train_size = len(self.splits["train"])
        self.val_size = len(self.splits["val"])
        self.my_core_model = Summarizer(self.tensor_device, self.hparams)
        if hparams.n_jobs_for_mcts == -1:
            self.n_processes = os.cpu_count()
        else:
            self.n_processes = hparams.n_jobs_for_mcts
        self.pool = mp.Pool(processes=self.n_processes)

    def __my_document_level_encoding(self, contents):
        """
        input:
            contents : list of output berttokenizer
        """
        sent_scores, mask_cls = self.my_core_model(contents)
        return sent_scores, mask_cls

    def forward(self, batch, subset):
        ids, contents, abstracts, raw_contents, raw_abstracts, scorers = batch
        batch_size = len(ids)

        contents_extracted, valid_sentences = self.__my_document_level_encoding(contents)

        masked_predictions = contents_extracted + valid_sentences.float().log()
        _, greedy_idxs = torch.topk(masked_predictions, self.n_sents_per_summary, sorted=False)

        sentence_gap = contents["sentence_gap"]
        true_greedy_idxs = greedy_idxs.clone()
        for sentence_gap_, true_greedy_idx in zip(sentence_gap, true_greedy_idxs):
            true_greedy_idx[0] += sentence_gap_[true_greedy_idx[0]]
            true_greedy_idx[1] += sentence_gap_[true_greedy_idx[1]]
            true_greedy_idx[2] += sentence_gap_[true_greedy_idx[2]]

        if subset == "train":
            greedy_rewards = []
            for scorer, sent_idxs in zip(scorers, true_greedy_idxs):
                greedy_rewards.append(scorer(sent_idxs.tolist()))
            greedy_rewards = torch.tensor(greedy_rewards)
        else:
            greedy_rewards = scorers.get_scores(true_greedy_idxs, raw_contents, raw_abstracts)

        targets = torch.zeros_like(contents_extracted)
        targets[:, :3] = 1

        loss = self.criterion(contents_extracted, targets) * valid_sentences
        loss = loss.sum(-1) / valid_sentences.sum(-1)
        loss = loss.mean()

        predicted_idxs_repart = torch.zeros(batch_size, 50, device=self.tensor_device)
        predicted_idxs_repart.scatter_(1, greedy_idxs, 1)

        self.predicted_idxs_repart[subset] += predicted_idxs_repart.sum(0)

        self.target_idxs_repart[subset] += F.pad(targets.sum(0), (0, 50 - targets.shape[-1]))

        return greedy_rewards, loss

    def training_step(self, batch, batch_idx):
        self.my_core_model.train()
        start = time.time()
        greedy_rewards, loss = self.forward(batch, subset="train")
        end = time.time()

        log_dict = {
            "greedy_rouge_mean": greedy_rewards.mean(),
            "loss": loss.detach(),
            "batch_time": end - start,
        }

        for key, val in log_dict.items():
            self.log(key, val, prog_bar="mean" in key)

        return loss

    def training_epoch_end(self, outputs):
        for i, idx_repart in enumerate(self.predicted_idxs_repart["train"] / self.train_size):
            self.log(f"predicted_idx_train_sent_{i}_epoch_{self.n_epochs_done}", idx_repart)

        self.predicted_idxs_repart["train"] = torch.zeros_like(self.predicted_idxs_repart["train"])

        for i, idx_repart in enumerate(self.target_idxs_repart["train"] / self.train_size):
            self.log(f"target_idx_train_sent_{i}_epoch_{self.n_epochs_done}", idx_repart)

        self.target_idxs_repart["train"] = torch.zeros_like(self.target_idxs_repart["train"])

    def validation_step(self, batch, batch_idx):
        self.my_core_model.eval()
        greedy_rewards, loss = self.forward(batch, subset="val")

        reward_dict = {"val_greedy_rouge_mean": greedy_rewards.mean(-1), "val_loss": loss.detach()}

        for name, val in reward_dict.items():
            self.log(name, val, prog_bar="loss" in name)

        return reward_dict["val_greedy_rouge_mean"].mean()

    def validation_epoch_end(self, outputs):
        if self.first_val_batches_done:
            for i, idx_repart in enumerate(self.predicted_idxs_repart["val"] / self.val_size):
                self.log(f"predicted_idx_val_sent_{i}_epoch_{self.n_epochs_done}", idx_repart)

            for i, idx_repart in enumerate(self.target_idxs_repart["val"] / self.val_size):
                self.log(f"target_idx_val_sent_{i}_epoch_{self.n_epochs_done}", idx_repart)

            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr)
        else:
            self.first_val_batches_done = True

        self.predicted_idxs_repart["val"] = torch.zeros_like(self.predicted_idxs_repart["val"])
        self.target_idxs_repart["val"] = torch.zeros_like(self.target_idxs_repart["val"])

    def test_step(self, batch, batch_idx):
        self.my_core_model.eval()
        greedy_rewards, loss = self.forward(batch, subset="test")

        reward_dict = {"test_greedy_rouge_mean": greedy_rewards.mean(-1), "test_loss": loss.detach()}

        for name, val in reward_dict.items():
            self.log(name, val)

    def test_epoch_end(self, outputs):
        for i, idx_repart in enumerate(self.predicted_idxs_repart["test"] / self.test_size):
            self.log(f"predicted_idx_test_sent_{i}", idx_repart)

        self.predicted_idxs_repart["test"] = torch.zeros_like(self.predicted_idxs_repart["test"])

        for i, idx_repart in enumerate(self.target_idxs_repart["test"] / self.test_size):
            self.log(f"target_idx_test_sent_{i}", idx_repart)

        self.target_idxs_repart["test"] = torch.zeros_like(self.target_idxs_repart["test"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{"params": self.my_core_model.parameters(), "lr": self.learning_rate}],
            lr=self.learning_rate,
            betas=[0, 0.999],
            weight_decay=self.weight_decay,
        )

        self.lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.2, verbose=True)

        return optimizer

    def train_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.colname_2_field_objs, self.reward_builder, subset="train"),
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
            collate_fn=TextDataCollator(self.colname_2_field_objs, self.reward_builder, subset="val"),
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        dataset = self.splits["test"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.colname_2_field_objs, self.reward_builder, subset="test"),
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return BertBinarySumLead3(dataset, reward, config)
