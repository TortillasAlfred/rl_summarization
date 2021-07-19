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

from src.domain.loader_utils import TextDataCollator
from .bertsum_transformer import Summarizer


class BertBinaryModel(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super().__init__()
        self.tensor_device = "cuda" if hparams.gpus > 0 and torch.cuda.is_available() else "cpu"

        self.hparams = hparams
        self.colname_2_field_objs = dataset.fields

        self.pad_idx = dataset.pad_idx
        self.splits = dataset.get_splits()
        self.reward_builder = reward
        self.n_epochs_done = 0

        self.train_batch_size = hparams.train_batch_size
        self.num_workers = hparams.num_workers
        self.test_batch_size = hparams.test_batch_size
        self.hidden_dim = hparams.hidden_dim
        self.decoder_dim = hparams.decoder_dim
        self.learning_rate = hparams.learning_rate
        self.n_sents_per_summary = hparams.n_sents_per_summary
        self.c_puct = hparams.c_puct
        self.weight_decay = hparams.weight_decay
        self.batch_idx = 0
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.idxs_repart = torch.zeros(50, dtype=torch.float32, device=self.tensor_device)
        self.test_size = len(self.splits["test"])
        self.train_size = len(self.splits["train"])
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

    def forward(self, batch, subset):  # (data_batch, "train" or "test")
        ids, contents, abstracts, raw_contents, raw_abstracts, scorers = batch
        batch_size = len(ids)

        contents_extracted, valid_sentences = self.__my_document_level_encoding(contents)

        masked_predictions = contents_extracted + valid_sentences.float().log()  # Adds 0 if sentence is valid else -inf
        _, greedy_idxs = torch.topk(masked_predictions, self.n_sents_per_summary, sorted=False)

        # revert greedy_idx into origin index (before truncate)
        sentence_gap = contents["sentence_gap"]
        for sentence_gap_, greedy_idx in zip(sentence_gap, greedy_idxs):
            greedy_idx[0] += sentence_gap_[greedy_idx[0]]
            greedy_idx[1] += sentence_gap_[greedy_idx[1]]
            greedy_idx[2] += sentence_gap_[greedy_idx[2]]

        if subset == "train":
            greedy_rewards = []
            for scorer, sent_idxs in zip(scorers, greedy_idxs):
                greedy_rewards.append(scorer(sent_idxs.tolist()))
            greedy_rewards = torch.tensor(greedy_rewards)

            targets = torch.autograd.Variable(torch.zeros_like(contents_extracted), requires_grad=False).cuda()
            best_summ_idxs = []
            for sentence_gap_, scorer, valid_sentences_ in zip(sentence_gap, scorers, valid_sentences):
                available_sents = torch.arange(valid_sentences_.sum(-1)) + torch.tensor(sentence_gap_)
                available_sents = torch.combinations(available_sents, r=3)
                available_scores = [
                    torch.tensor([scorer(available_idxs.tolist())]) for available_idxs in available_sents
                ]
                available_scores = torch.stack(available_scores).squeeze()
                best_available_summ_idxs = available_sents[available_scores.sum(-1).argmax(0)]
                best_summ_idxs.append(torch.tensor(best_available_summ_idxs))

            best_summ_idxs = torch.stack(best_summ_idxs).cuda()
            targets = targets.scatter(-1, best_summ_idxs, 1)

            loss = self.criterion(contents_extracted, targets) * valid_sentences
            loss = loss.sum(-1) / valid_sentences.sum(-1)
            loss = loss.mean()

            suboptimality = (contents_extracted.max() - targets.max()).mean()

            return greedy_rewards, loss, suboptimality
        else:
            greedy_rewards = scorers.get_scores(greedy_idxs, raw_contents, raw_abstracts)

            if subset == "test":
                idxs_repart = torch.zeros(batch_size, 50, device=self.tensor_device)
                idxs_repart.scatter_(1, greedy_idxs, 1)

                self.idxs_repart += idxs_repart.sum(0)

            return torch.from_numpy(greedy_rewards) if greedy_rewards.ndim > 1 else torch.tensor([greedy_rewards])

    def training_step(self, batch, batch_idx):
        start = time.time()
        greedy_rewards, loss, suboptimality = self.forward(batch, subset="train")
        end = time.time()

        log_dict = {
            "greedy_rouge_mean": greedy_rewards.mean(),
            "suboptimality": suboptimality.mean(),
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

        return reward_dict["val_greedy_rouge_mean"].mean()

    def validation_epoch_end(self, outputs):
        mean_rouge = torch.stack(outputs).mean()
        self.lr_scheduler.step(mean_rouge)

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
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
        return BertBinaryModel(dataset, reward, config)
