from src.domain.loader_utils import TextDataCollator

import time
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .bertsum_transformer import Summarizer


class BertBanditSum(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super().__init__()

        self.tensor_device = "cuda" if hparams.gpus > 0 and torch.cuda.is_available() else "cpu"

        self.hparams = hparams
        self.colname_2_field_objs = dataset.fields
        self.pad_idx = dataset.pad_idx
        self.reward_builder = reward
        self.idxs_repart = torch.zeros(50, dtype=torch.float32, device=self.tensor_device)

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
        self.n_sents_per_summary = hparams.n_sents_per_summary
        self.weight_decay = hparams.weight_decay
        self.batch_idx = 0
        self.my_core_model = Summarizer(self.tensor_device, self.hparams)

    def __my_document_level_encoding(self, contents):
        """
        input:
            contents : list of output berttokenizer
        """
        sent_scores, mask_cls = self.my_core_model(contents)
        sent_scores = torch.sigmoid(sent_scores)
        return sent_scores, mask_cls

    def select_idxs(self, affinities, valid_sentences):
        n_docs = affinities.shape[0]

        affinities = affinities.repeat_interleave(self.n_repeats_per_sample, 0)
        valid_sentences = valid_sentences.repeat_interleave(self.n_repeats_per_sample, 0)

        all_idxs = torch.zeros(
            (n_docs, self.n_repeats_per_sample, self.n_sents_per_summary), dtype=torch.int, device=affinities.device
        )
        all_logits = torch.zeros(
            (n_docs, self.n_repeats_per_sample, self.n_sents_per_summary), dtype=torch.float, device=affinities.device
        )

        uniform_distro = torch.ones_like(affinities) * valid_sentences
        for step in range(self.n_sents_per_summary):
            step_affinities = affinities * valid_sentences
            step_affinities = step_affinities / step_affinities.sum(-1, keepdim=True)

            step_unif = uniform_distro * valid_sentences
            step_unif = step_unif / step_unif.sum(-1, keepdim=True)

            step_probas = self.epsilon * step_unif + (1 - self.epsilon) * step_affinities

            # I added this lines to prevent probability turning into 0
            step_probas = step_probas * step_probas.ge(0)

            c = Categorical(step_probas)

            idxs = c.sample()
            logits = c.logits.gather(1, idxs.unsqueeze(1))

            valid_sentences = valid_sentences.scatter(1, idxs.unsqueeze(1), 0)

            all_idxs[:, :, step] = idxs.view(n_docs, self.n_repeats_per_sample)
            all_logits[:, :, step] = logits.view(n_docs, self.n_repeats_per_sample)

        return all_idxs, all_logits.sum(-1)

    def forward(self, batch, subset):
        ids, contents, abstracts, raw_contents, raw_abstracts, scorers = batch
        batch_size = len(ids)

        contents_extracted, valid_sentences = self.__my_document_level_encoding(contents)
        masked_predictions = contents_extracted + valid_sentences.float().log()  # Adds 0 if sentence is valid else -inf
        _, greedy_idxs = torch.topk(masked_predictions, self.n_sents_per_summary, sorted=False)

        sentence_gap = contents["sentence_gap"]
        for sentence_gap_, greedy_idx in zip(sentence_gap, greedy_idxs):
            greedy_idx[0] += sentence_gap_[greedy_idx[0]]
            greedy_idx[1] += sentence_gap_[greedy_idx[1]]
            greedy_idx[2] += sentence_gap_[greedy_idx[2]]

        if subset == "train":
            greedy_rewards = []
            for scorer, sent_idxs in zip(scorers, greedy_idxs):
                greedy_rewards.append(scorer(sent_idxs.tolist()))
            greedy_rewards = torch.tensor(greedy_rewards).unsqueeze(1)

            selected_idxs, selected_logits = self.select_idxs(contents_extracted, valid_sentences)
            generated_rewards = []
            for scorer, batch_idxs in zip(scorers, selected_idxs):
                generated_rewards.append(torch.tensor([scorer(sent_idxs.tolist()) for sent_idxs in batch_idxs]))

            generated_rewards = torch.stack(generated_rewards)

            greedy_rewards = greedy_rewards.repeat_interleave(self.n_repeats_per_sample, 1)

            rewards = ((greedy_rewards - generated_rewards)).clone().detach().to(device=selected_logits.device)

            if len(rewards.shape) < 3:
                rewards = rewards.unsqueeze(-1)

            loss = rewards.mean(-1) * selected_logits
            loss = loss.mean()
            return generated_rewards, loss, greedy_rewards
        else:
            greedy_rewards = scorers.get_scores(greedy_idxs, raw_contents, raw_abstracts)

            if subset == "test":
                idxs_repart = torch.zeros(batch_size, 50, device=self.tensor_device)
                idxs_repart.scatter(1, greedy_idxs, 1)

                self.idxs_repart += idxs_repart.sum(0)

            return torch.from_numpy(greedy_rewards) if greedy_rewards.ndim > 1 else torch.tensor([greedy_rewards])

    def training_step(self, batch, batch_idx):
        self.my_core_model.train()
        start = time.time()
        generated_rewards, loss, greedy_rewards = self.forward(batch, subset="train")
        end = time.time()

        log_dict = {
            "greedy_rouge_mean": greedy_rewards.mean(),
            "generated_rouge_mean": generated_rewards.mean(),
            "loss": loss.detach(),
            "batch_time": end - start,
        }

        for key, val in log_dict.items():
            self.log(key, val, prog_bar="greedy" in key)

        return loss

    def validation_step(self, batch, batch_idx):
        self.my_core_model.eval()
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
        self.my_core_model.eval()
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
        return BertBanditSum(dataset, reward, config)
