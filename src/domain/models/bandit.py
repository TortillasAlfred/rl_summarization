import pytorch_lightning as pl

import logging
import torch
from torch.utils.data import Subset, DataLoader
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
import numpy as np
from torchtext.data import BucketIterator


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
            num_layers=1,
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
        pad_tokens = ~(contents == self.pad_idx)
        contents = self.embeddings(contents)
        b_size, n_sents, n_tokens, emb_dim = contents.shape
        tokens_mask = pad_tokens.sum(-1).unsqueeze(-1)
        sentences_mask = tokens_mask == 0
        contents = contents.reshape(-1, n_tokens, emb_dim)
        contents, _ = self.wl_encoder(contents)
        contents = contents.reshape(b_size, n_sents, n_tokens, -1)
        contents = contents.mean(-2) / tokens_mask
        contents[sentences_mask.expand_as(contents)] = 0
        return contents, ~sentences_mask.squeeze(-1)

    def __sentence_level_encoding(self, contents):
        contents, _ = self.sl_encoder(contents)
        return contents

    def __decoding(self, contents):
        b_size, n_sents, input_dim = contents.shape
        contents = contents.reshape(-1, input_dim)
        contents = self.decoder(contents)
        return contents.reshape(b_size, n_sents)

    def __produce_affinities(self, contents):
        contents, sentences_mask = self.__word_level_encoding(contents)
        contents = self.__sentence_level_encoding(contents)
        contents = self.__decoding(contents)
        contents = contents * sentences_mask

        return contents, sentences_mask

    def select_idxs(self, affinities, available_sentences_mask):
        affinities = affinities.expand(self.n_repeats_per_sample, -1, -1).permute(
            1, 0, 2
        )
        available_sentences_mask = available_sentences_mask.expand(
            self.n_repeats_per_sample, -1, -1
        ).permute(1, 0, 2)
        sample_size = affinities.shape[:-1]
        idxs = torch.zeros(
            (sample_size[0], self.n_repeats_per_sample, self.n_sents_per_summary),
            dtype=torch.int,
            device=affinities.device,
        )
        logits = torch.zeros(
            (sample_size[0], self.n_repeats_per_sample, self.n_sents_per_summary),
            dtype=torch.float,
            device=affinities.device,
        )
        uniform_sampler = Uniform(0, 1)
        for i in range(self.n_sents_per_summary):
            exploring_idxs = uniform_sampler.rsample(sample_size)
            exploring_idxs = exploring_idxs < self.epsilon
            masked_affinities = affinities.clone()
            masked_affinities[exploring_idxs, :] = 1
            masked_affinities = masked_affinities * available_sentences_mask
            dist = Categorical(probs=masked_affinities)
            retained_idxs = dist.sample().unsqueeze(-1)
            probs = dist.probs
            retained_probs = probs.gather(-1, retained_idxs).squeeze(-1)
            logit = (
                self.epsilon / available_sentences_mask.sum(-1).float()
                + (1 - self.epsilon) * retained_probs
            ).log()
            logits[:, :, i] = logit
            idxs[:, :, i] = retained_idxs.squeeze(-1)
            available_sentences_mask = available_sentences_mask.scatter(
                -1, retained_idxs, 0
            )

        return idxs, logits.sum(-1)

    def forward(self, input_texts, train=False):
        raw_contents, contents = input_texts
        affinities, sentences_mask = self.__produce_affinities(contents)
        greedy_idxs = affinities.argsort(descending=True)[:, : self.n_sents_per_summary]
        # for idxs, content in zip(greedy_idxs, raw_contents):
        #     logging.info(f"***CONTENT LEN***:{len(content)}")
        #     logging.info(f"max idx attempt: {max(idxs.tolist())}")

        greedy_summaries = [
            [content[i] for i in idxs]
            for content, idxs in zip(raw_contents, greedy_idxs)
        ]

        if train:
            idxs, logits = self.select_idxs(affinities, sentences_mask)

            # for batch, content in zip(idxs, raw_contents):
            #     logging.info(f"***CONTENT LEN***:{len(content)}")
            #     for idx in batch:
            #         logging.info(f"max idx attempt: {max(idx.tolist())}")

            return (
                greedy_summaries,
                [
                    [content[i] for i in idx]
                    for content, batch in zip(raw_contents, idxs)
                    for idx in batch
                ],
                logits,
            )
        else:
            return greedy_summaries

    def training_step(self, batch, batch_idx):
        input_texts, (ref_summaries, _) = batch

        greedy_summaries, hyp_summaries, logits = self.forward(input_texts, train=True)

        n_hyps = len(hyp_summaries)
        all_summaries = hyp_summaries + greedy_summaries
        all_rewards = self.reward(all_summaries, ref_summaries, input_texts[1].device)
        rewards = all_rewards[:n_hyps].view(
            -1, self.n_repeats_per_sample, self.n_sents_per_summary
        )
        greedy_rewards = all_rewards[n_hyps:].unsqueeze(1)

        # loss = (greedy_rewards - rewards).mean(-1) * logits
        loss = -rewards.mean(-1) * logits
        loss = loss.mean()

        return {
            "train_loss": loss,
            "train_reward": rewards.mean((0, 1)),
            "train_greedy_reward": greedy_rewards.mean((0, 1)),
        }

    def training_end(self, outputs):
        train_loss = outputs["train_loss"]
        train_reward = outputs["train_reward"]
        train_greedy_reward = outputs["train_greedy_reward"]

        tqdm_dict = {
            "train_reward": train_reward.mean().item(),
            "train_greedy_reward": train_greedy_reward.mean().item(),
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

    def train_dataloader(self):
        logging.info("training data loader called")
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
        logging.info("val data loader called")
        return BucketIterator(
            self.splits["val"],
            train=False,
            batch_size=self.test_batch_size,
            sort=False,
            device=self.embeddings.weight.device,
        )

    @pl.data_loader
    def test_dataloader(self):
        logging.info("test data loader called")
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

