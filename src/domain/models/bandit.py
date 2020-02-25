import pytorch_lightning as pl

import logging
import torch
from torch.utils.data import Subset, DataLoader
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
import numpy as np
from torchtext.data import BucketIterator
import torch.nn.functional as F


class BanditSum(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super(BanditSum, self).__init__()
        self.dataset = dataset
        self.reward = reward
        self.embedding_dim = self.dataset.embedding_dim
        self.pad_idx = self.dataset.pad_idx
        self.splits = self.dataset.get_splits()
        self.n_epochs_done = 0

        self.items_per_epoch = hparams["items_per_epoch"]
        self.train_batch_size = hparams["train_batch_size"]
        self.test_batch_size = hparams["test_batch_size"]
        self.hidden_dim = hparams["hidden_dim"]
        self.decoder_dim = hparams["decoder_dim"]
        self.n_repeats_per_sample = hparams["n_repeats_per_sample"]
        self.learning_rate = hparams["learning_rate"]
        self.epsilon = hparams["epsilon"]
        self.n_sents_per_summary = hparams["n_sents_per_summary"]

        self.__build_model(hparams["hidden_dim"], hparams["decoder_dim"])

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
        )
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

    def __word_level_encoding(self, contents):
        valid_tokens = ~(contents == self.pad_idx)
        sentences_len = valid_tokens.sum(dim=-1)
        valid_sentences = sentences_len > 0
        contents = self.embeddings(contents)
        contents = torch.cat([self.wl_encoder(content)[0] for content in contents])
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

    def __produce_affinities(self, contents):
        contents, valid_sentences = self.__word_level_encoding(contents)
        contents = self.__sentence_level_encoding(contents)
        contents = self.__decoding(contents)
        contents = contents * valid_sentences

        return contents, valid_sentences

    def select_idxs(self, affinities, valid_sentences):
        n_docs = affinities.shape[0]
        idxs = torch.zeros(
            (n_docs, self.n_repeats_per_sample, self.n_sents_per_summary),
            dtype=torch.int,
            device=affinities.device,
        )
        logits = torch.zeros(
            (n_docs, self.n_repeats_per_sample, self.n_sents_per_summary),
            dtype=torch.float,
            device=affinities.device,
        )
        uniform_sampler = Uniform(0, 1)
        for doc in range(n_docs):
            for repeat in range(self.n_repeats_per_sample):
                probs = affinities[doc].clone()
                val_sents = valid_sentences[doc].clone()
                for sent in range(self.n_sents_per_summary):
                    probs = F.softmax(probs, dim=-1)
                    probs = probs * val_sents
                    probs = probs / probs.sum()
                    if uniform_sampler.sample() <= self.epsilon:
                        idx = Categorical(val_sents.float()).sample()
                    else:
                        idx = Categorical(probs).sample()
                    logit = (
                        self.epsilon / val_sents.sum()
                        + (1 - self.epsilon) * probs[idx] / probs.sum()
                    ).log()
                    val_sents = val_sents.clone()
                    val_sents[idx] = 0
                    idxs[doc, repeat, sent] = idx
                    logits[doc, repeat, sent] = logit

        return idxs, logits.sum(-1)

    def forward(self, input_texts, train=False):
        raw_contents, contents = input_texts
        affinities, valid_sentences = self.__produce_affinities(contents)
        greedy_idxs = affinities.argsort(descending=True)[:, : self.n_sents_per_summary]

        greedy_summaries = [
            [content[i] for i in idxs if i < len(content)]
            for content, idxs in zip(raw_contents, greedy_idxs)
        ]

        if train:
            idxs, logits = self.select_idxs(affinities, valid_sentences)

            return (
                greedy_summaries,
                [
                    [content[i] for i in idx if i < len(content)]
                    for content, batch in zip(raw_contents, idxs)
                    for idx in batch
                ],
                logits,
                affinities,
                greedy_idxs,
                idxs,
            )
        else:
            return greedy_summaries

    def training_step(self, batch, batch_idx):
        input_texts, (ref_summaries, _) = batch

        (
            greedy_summaries,
            hyp_summaries,
            logits,
            affinities,
            greedy_idxs,
            idxs,
        ) = self.forward(input_texts, train=True)

        n_hyps = len(hyp_summaries)
        all_summaries = hyp_summaries + greedy_summaries
        all_rewards = self.reward(all_summaries, ref_summaries, input_texts[1].device)
        rewards = all_rewards[:n_hyps].view(-1, self.n_repeats_per_sample, 3)
        greedy_rewards = all_rewards[n_hyps:].unsqueeze(1)

        # r = (rewards - greedy_rewards).mean(-1)
        r = rewards.mean(-1)
        loss = -r * logits
        loss = loss.mean()

        # y = torch.zeros_like(logits)
        # y[:, : self.n_sents_per_summary] = 1
        # loss = F.mse_loss(logits, y)

        return {
            "train_loss": loss,
            "train_reward": rewards.mean((0, 1)),
            "train_greedy_reward": greedy_rewards.mean((0, 1)),
            "max_prob": affinities.max(),
        }

    def training_end(self, outputs):
        train_loss = outputs["train_loss"]
        train_reward = outputs["train_reward"]
        train_greedy_reward = outputs["train_greedy_reward"]
        max_prob = outputs["max_prob"]

        tqdm_dict = {
            "loss": train_loss.item(),
            "train_reward": train_reward.mean().item(),
            "train_greedy_reward": train_greedy_reward.mean().item(),
            "max_prob": max_prob.item(),
        }

        # show train_loss and train_acc in progress bar but only log train_loss
        results = {
            "loss": train_loss,
            "progress_bar": tqdm_dict,
            "log": {
                "train_loss": train_loss.item(),
                "train_r1": train_reward[0].item(),
                "train_r2": train_reward[1].item(),
                "train_rL": train_reward[2].item(),
                "train_reward": train_reward.mean().item(),
                "train_greedy_r1": train_greedy_reward[0].item(),
                "train_greedy_r2": train_greedy_reward[1].item(),
                "train_greedy_rL": train_greedy_reward[2].item(),
                "train_greedy_reward": train_greedy_reward.mean().item(),
                "max_prob": max_prob.item(),
            },
        }

        return results

    def validation_step(self, batch, batch_idx):
        input_texts, (ref_summaries, _) = batch

        greedy_summaries = self.forward(input_texts)

        greedy_rewards = self.reward(
            greedy_summaries, ref_summaries, input_texts[1].device
        )

        return {
            "val_greedy_reward": greedy_rewards.mean(0),
        }

    def test_step(self, batch, batch_idx):
        input_texts, (ref_summaries, _) = batch

        greedy_summaries = self.forward(input_texts)

        greedy_rewards = self.reward(
            greedy_summaries, ref_summaries, input_texts[1].device
        )

        return {
            "test_greedy_reward": greedy_rewards.mean(0),
        }

    def validation_end(self, outputs):
        val_greedy_reward_mean = 0

        for output in outputs:
            val_greedy_reward_mean += output["val_greedy_reward"]

        val_greedy_reward_mean /= len(outputs)

        tqdm_dict = {
            "val_greedy_reward": val_greedy_reward_mean.mean().item(),
        }

        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            "val_loss": -val_greedy_reward_mean.mean(),
            "progress_bar": tqdm_dict,
            "log": {
                "log": {
                    "val_greedy_r1": val_greedy_reward_mean[0].item(),
                    "val_greedy_r2": val_greedy_reward_mean[1].item(),
                    "val_greedy_rL": val_greedy_reward_mean[2].item(),
                    "val_greedy_reward": val_greedy_reward_mean.mean().item(),
                },
            },
        }

        return results

    def test_end(self, outputs):
        test_greedy_reward_mean = 0

        for output in outputs:
            test_greedy_reward_mean += output["test_greedy_reward"]

        test_greedy_reward_mean /= len(outputs)

        tqdm_dict = {
            "test_greedy_reward": test_greedy_reward_mean.mean().item(),
        }

        # show test_loss and test_acc in progress bar but only log test_loss
        results = {
            "test_loss": -test_greedy_reward_mean.mean(),
            "progress_bar": tqdm_dict,
            "log": {
                "test_greedy_r1": test_greedy_reward_mean[0].item(),
                "test_greedy_r2": test_greedy_reward_mean[1].item(),
                "test_greedy_rL": test_greedy_reward_mean[2].item(),
                "test_greedy_reward": test_greedy_reward_mean.mean().item(),
            },
        }

        return results

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
            weight_decay=1e-6,
        )
        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        train_split = self.splits["train"]
        n_items = min(self.items_per_epoch, len(train_split))
        begin_idx = self.n_epochs_done * n_items
        end_idx = begin_idx + n_items
        indices = np.arange(begin_idx, end_idx) % len(train_split)
        train_set = Subset(train_split, indices)
        train_set.sort_key = None
        train_set.fields = train_split.fields
        self.n_epochs_done += 1
        return BucketIterator(
            train_set,
            train=True,
            batch_size=self.train_batch_size,
            sort=False,
            device=self.embeddings.weight.device,
        )

    @pl.data_loader
    def val_dataloader(self):
        return BucketIterator(
            self.splits["val"],
            train=False,
            batch_size=self.test_batch_size,
            sort=False,
            device=self.embeddings.weight.device,
        )

    @pl.data_loader
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
        return BanditSum(dataset, reward, config,)
