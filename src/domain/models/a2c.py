from src.domain.rollouts import Rollouts

import pytorch_lightning as pl

import logging
import torch
from torch.utils.data import Subset, DataLoader
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
import numpy as np
from torchtext.data import BucketIterator
import torch.nn.functional as F
from collections import deque


class A2C(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super(A2C, self).__init__()
        self.dataset = dataset
        self.reward = reward
        self.embedding_dim = self.dataset.embedding_dim
        self.pad_idx = self.dataset.pad_idx
        self.splits = self.dataset.get_splits()
        self.n_epochs_done = 0

        self.items_per_epoch = hparams["items_per_epoch"]
        self.train_batch_size = hparams["train_batch_size"]
        self.test_batch_size = hparams["test_batch_size"]
        self.n_repeats_per_sample = hparams["n_repeats_per_sample"]
        self.learning_rate = hparams["learning_rate"]
        self.n_sents_per_summary = hparams["n_sents_per_summary"]

        self.rewards_buffer = deque(maxlen=25)

        self.__build_model(hparams["hidden_dim"])

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
        )
        self.sl_encoder = torch.nn.LSTM(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 1),
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 1),
        )

    def _compute_sent_lens(self, contents):
        pad_tokens = contents == self.pad_idx
        sent_lens = (~pad_tokens).sum(-1)
        return sent_lens

    def extract_features(self, contents):
        sent_lens = self._compute_sent_lens(contents)

        valid_sents = sent_lens > 0
        x = self.embeddings(contents).sum(-2)
        x, _ = self.wl_encoder(x)
        sent_features = torch.zeros_like(x)
        sent_features[valid_sents] = x[valid_sents] / sent_lens[valid_sents].unsqueeze(
            -1
        )
        sent_features, (_, doc_features) = self.sl_encoder(sent_features)
        doc_features = doc_features.view(contents.shape[0], -1)

        return sent_features, doc_features, valid_sents

    def get_policy_dist(
        self, sent_features, doc_features, valid_sentences, selected_idxs
    ):
        logits = self.policy(sent_features).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        # probs = probs * ~selected_idxs * valid_sentences
        probs = probs * valid_sentences

        probs = probs / probs.sum(-1, keepdim=True)

        return Categorical(probs=probs), probs

    def get_values(self, sent_features, doc_features, selected_idxs):
        n_repeats, batch_size, n_sents = sent_features.shape[:3]

        # summ_features = sent_features * selected_idxs.unsqueeze(-1)
        summ_features = sent_features
        summ_features = summ_features.view(n_repeats * batch_size, n_sents, -1)
        _, (_, summ_features) = self.sl_encoder(summ_features)
        summ_features = summ_features.view(n_repeats * batch_size, -1)
        summ_features = summ_features.view(n_repeats, batch_size, -1)

        return self.value(torch.cat((summ_features, doc_features), dim=-1)).squeeze(-1)

    def act(
        self, sent_features, doc_features, valid_sentences, selected_idxs, done_masks
    ):
        dists, prob = self.get_policy_dist(
            sent_features, doc_features, valid_sentences, selected_idxs
        )
        vals = self.get_values(sent_features, doc_features, selected_idxs)

        idxs = dists.sample()
        logprobs = dists.log_prob(idxs)
        entropies = dists.entropy()

        # Apply done masks
        # vals = vals * done_masks
        # idxs = idxs * done_masks
        # logprobs = logprobs * done_masks
        # entropies = entropies * done_masks

        return vals, idxs, logprobs, entropies, prob

    def compute_rollouts(self, contents, num_repeats):
        sent_features, doc_features, valid_sentences = self.extract_features(contents)

        n_texts, n_sents, _ = contents.shape
        rollouts = Rollouts(
            n_texts=n_texts,
            n_sents=n_sents,
            n_steps=self.n_sents_per_summary,
            n_repeats=num_repeats,
            device=contents.device,
        )

        sent_features = sent_features.repeat(num_repeats, 1, 1, 1)
        doc_features = doc_features.repeat(num_repeats, 1, 1)
        valid_sentences = valid_sentences.repeat(num_repeats, 1, 1)
        probs = []

        for step in range(self.n_sents_per_summary):
            selected_idxs = rollouts.selected_idxs[step]

            vals, idxs, logprobs, entropies, prob = self.act(
                sent_features,
                doc_features,
                valid_sentences,
                selected_idxs,
                rollouts.done_masks[step],
            )

            done_masks = selected_idxs.sum(-1) < valid_sentences.sum(-1)

            rollouts.store(
                selected_idxs.scatter(-1, idxs.unsqueeze(-1), 1),
                vals,
                logprobs,
                entropies,
                done_masks,
                idxs,
            )

            probs.append(prob)

        next_values = self.get_values(sent_features, doc_features, selected_idxs)
        next_values = next_values * rollouts.done_masks[-1]

        return rollouts, next_values, torch.stack(probs)

    def _compute_rewards(self, idxs, done_masks, raw_contents, ref_summaries):
        hyp_summaries = []
        formatted_refs = []

        for i, step in enumerate(idxs):
            step_summaries = []
            for j, fold in enumerate(step):
                fold_summaries = []
                for k, (text, content) in enumerate(zip(fold, raw_contents)):
                    summary = [content[text]]

                    if i > 0:
                        summary = hyp_summaries[i - 1][j][k] + [content[text]]

                    fold_summaries.append(summary)
                    formatted_refs.append(ref_summaries[k])
                step_summaries.append(fold_summaries)
            hyp_summaries.append(step_summaries)

        hyp_summaries = [
            summ for step in hyp_summaries for fold in step for summ in fold
        ]

        rewards = self.reward(hyp_summaries, formatted_refs, idxs.device)
        rewards = rewards.view(*idxs.shape, -1)
        rewards = rewards * done_masks[:-1].unsqueeze(-1)

        return rewards

    def training_step(self, batch, batch_idx):
        (raw_contents, contents), (ref_summaries, _) = batch

        rollouts, next_values, probs = self.compute_rollouts(
            contents, num_repeats=self.n_repeats_per_sample
        )

        rewards = self._compute_rewards(
            rollouts.idxs, rollouts.done_masks, raw_contents, ref_summaries
        )

        rollouts.compute_returns(rewards.mean(-1), next_values, gamma=0.99)

        advantages = rollouts.returns[:-1] - rollouts.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        action_loss = (-advantages.detach() * rollouts.logprobs).mean()
        value_loss = F.smooth_l1_loss(rollouts.values, rollouts.returns[:-1])
        entropy_loss = rollouts.entropies.mean()

        loss = action_loss + 0.5 * value_loss + 0.01 * entropy_loss

        # TODO : Logging

        return {
            "train_loss": loss,
            "final_reward": rewards[-1].mean((0, 1)),
        }

    def training_end(self, outputs):
        train_loss = outputs["train_loss"]
        final_reward = outputs["final_reward"]

        self.rewards_buffer.append(final_reward.cpu().numpy())

        tqdm_dict = {
            "loss": train_loss.item(),
            "running_reward": np.mean(self.rewards_buffer),
            "last_reward": final_reward.mean(),
        }

        # show train_loss and train_acc in progress bar but only log train_loss
        results = {
            "loss": train_loss,
            "progress_bar": tqdm_dict,
            "log": {
                "train_loss": train_loss.item(),
                "running_reward": np.mean(self.rewards_buffer),
                "last_reward": final_reward.mean(),
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
                {"params": self.policy.parameters()},
                {"params": self.value.parameters()},
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

    @staticmethod
    def from_config(dataset, reward, config):
        return A2C(dataset, reward, config,)
