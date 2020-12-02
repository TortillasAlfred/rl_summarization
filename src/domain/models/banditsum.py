from src.domain.loader_utils import text_data_collator


import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
import numpy as np
from torchtext.data import BucketIterator
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict, namedtuple


class BanditSum(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super(BanditSum, self).__init__()
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
        self.n_sents_per_summary = hparams.n_sents_per_summary
        self.dropout = hparams.dropout
        self.weight_decay = hparams.weight_decay
        self.batch_idx = 0

        self.__build_model()
        self.model = RLSummModel(hparams.hidden_dim, hparams.decoder_dim, self.dropout,)

    def __build_model(self):
        self.embeddings = torch.nn.Embedding.from_pretrained(
            self.dataset.vocab.vectors, freeze=False, padding_idx=self.pad_idx
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
        sent_contents, doc_contents = self.model.sentence_level_encoding(contents)
        affinities = self.model.produce_affinities(sent_contents)
        affinities = affinities * valid_sentences

        return Categorical(probs=affinities), affinities, valid_sentences

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

    def forward(self, batch, subset):
        raw_contents, contents, raw_abstracts, abstracts, ids, scorers = batch.values()
        batch_size = len(contents)

        self.wl_encoder.flatten_parameters()
        self.model.sl_encoder.flatten_parameters()

        action_dist, action_vals, valid_sentences = self.__extract_features(contents)

        _, greedy_idxs = torch.topk(
            action_dist.probs, self.n_sents_per_summary, sorted=False
        )
        greedy_rewards = []
        for scorer, sent_idxs in zip(scorers, greedy_idxs):
            greedy_rewards.append(
                torch.from_numpy(scorer.get_score(sent_idxs.tolist()))
            )
        greedy_rewards = torch.stack(greedy_rewards)

        if subset == "train":
            selected_idxs, selected_logits = self.select_idxs(
                action_vals, valid_sentences
            )

            generated_rewards = []
            for scorer, batch_idxs in zip(scorers, selected_idxs):
                generated_rewards.append(
                    torch.stack(
                        [
                            torch.from_numpy(scorer.get_score(sent_idxs.tolist()))
                            for sent_idxs in batch_idxs
                        ]
                    )
                )
            generated_rewards = torch.stack(generated_rewards)

            greedy_rewards = greedy_rewards.unsqueeze(1).repeat(
                1, self.n_repeats_per_sample, 1
            )

            rewards = (
                (generated_rewards - greedy_rewards)
                .mean(-1)
                .clone()
                .detach()
                .to(device=selected_logits.device)
            )
            loss = -rewards * selected_logits
            loss = loss.mean()

            return generated_rewards, loss, greedy_rewards
        else:
            return greedy_rewards

    def get_step_output(self, loss, greedy_rewards, generated_rewards):
        output_dict = {}

        log_dict = {
            "greedy_rouge_1": greedy_rewards[:, :, 0].mean(),
            "greedy_rouge_2": greedy_rewards[:, :, 1].mean(),
            "greedy_rouge_L": greedy_rewards[:, :, 2].mean(),
            "greedy_rouge_mean": greedy_rewards.mean(-1).mean(),
            "generated_rouge_1": generated_rewards[:, :, 0].mean(),
            "generated_rouge_2": generated_rewards[:, :, 1].mean(),
            "generated_rouge_L": generated_rewards[:, :, 2].mean(),
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
        greedy_rewards = self.forward(batch, subset="test")

        reward_dict = {
            "val_greedy_rouge_1": greedy_rewards[:, 0],
            "val_greedy_rouge_2": greedy_rewards[:, 1],
            "val_greedy_rouge_L": greedy_rewards[:, 2],
            "val_greedy_rouge_mean": greedy_rewards.mean(-1),
        }

        return reward_dict

    def test_step_end(self, outputs):
        for vals in outputs.values():
            vals = vals.mean()

        return outputs

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
        return self.generic_epoch_end(outputs, is_test=True)

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
            ],
            lr=self.learning_rate,
            betas=[0, 0.999],
            weight_decay=self.weight_decay,
        )

        self.lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=6, factor=0.1, verbose=True
        )

        return optimizer

    def train_dataloader(self):
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=text_data_collator(
                dataset.fields, self.reward_builder, subset="train"
            ),
            batch_size=self.train_batch_size,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = self.splits["val"]
        return DataLoader(
            dataset,
            collate_fn=text_data_collator(
                dataset.fields, self.reward_builder, subset="val"
            ),
            batch_size=self.test_batch_size,
            num_workers=16,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = self.splits["test"]
        return DataLoader(
            dataset,
            collate_fn=text_data_collator(
                dataset.fields, self.reward_builder, subset="test"
            ),
            batch_size=self.test_batch_size,
            num_workers=16,
            pin_memory=True,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return BanditSum(dataset, reward, config,)


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
            torch.nn.Linear(hidden_dim * 2, decoder_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, 1),
            torch.nn.Sigmoid(),
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

    def produce_affinities(self, sent_contents):
        affinities = self.decoder(sent_contents).squeeze(-1)

        return affinities

    def get_sents_from_summs(self, sent_contents, sampled_summs):
        all_sents = []

        for sents_doc, sampled_sents in zip(sent_contents, sampled_summs):
            for summ in sampled_sents:
                all_sents.append(torch.cat([sents_doc[sent_id] for sent_id in summ]))

        return torch.stack(all_sents)

    def forward(self, x):
        pass
