from src.domain.loader_utils import TextDataCollator

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import combinations
import random
import os


class LinSIT(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super(LinSIT, self).__init__()
        self.reward_builder = reward
        self.fields = dataset.fields

        self.embedding_dim = dataset.embedding_dim
        self.pad_idx = dataset.pad_idx
        self.splits = dataset.get_splits()
        self.n_epochs_done = 0

        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size
        self.hidden_dim = hparams.hidden_dim
        self.decoder_dim = hparams.decoder_dim
        self.n_repeats_per_sample = hparams.n_repeats_per_sample
        self.learning_rate = hparams.learning_rate
        self.epsilon = hparams.epsilon
        self.epsilon_min = hparams.epsilon_min
        self.epsilon_decay = hparams.epsilon_decay
        self.n_sents_per_summary = hparams.n_sents_per_summary
        self.dropout = hparams.dropout
        self.weight_decay = hparams.weight_decay
        self.pretraining_path = hparams.pretraining_path
        self.batch_idx = 0

        os.makedirs(self.pretraining_path, exist_ok=True)

        self.__build_model(dataset)
        self.model = RLSummModel(hparams.hidden_dim, hparams.decoder_dim, self.dropout,)

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

    def warmup_oful(self, valid_sentences, scorers):
        all_sampled_summs = []
        all_scores = []

        for valid_sents, scorer in zip(valid_sentences, scorers):
            all_sums = list(combinations(list(range(valid_sents.sum())), 3))
            sampled_summs = np.random.choice(len(all_sums), 64, replace=True)
            sampled_summs = [list(all_sums[summ]) for summ in sampled_summs]
            scores = torch.tensor(
                [scorer.scores[tuple(summ)].mean() for summ in sampled_summs],
                device=self.device,
            )
            all_scores.append(scores)
            all_sampled_summs.append(sampled_summs)

        all_scores = torch.cat(all_scores)

        return all_sampled_summs, all_scores

    def forward(self, batch, subset):
        raw_contents, contents, raw_abstracts, abstracts, ids, scorers = batch.values()
        batch_size = len(contents)

        self.wl_encoder.flatten_parameters()
        self.model.sl_encoder.flatten_parameters()

        (action_vals, valid_sentences, sent_contents,) = self.__extract_features(
            contents
        )

        _, greedy_idxs = torch.topk(action_vals, self.n_sents_per_summary, sorted=False)
        greedy_rewards = []
        for scorer, sent_idxs in zip(scorers, greedy_idxs):
            greedy_rewards.append(
                torch.from_numpy(scorer.get_score(sent_idxs.tolist()))
            )
        greedy_rewards = torch.stack(greedy_rewards)

        if subset == "train":
            self.batch_idx += 1

            if self.batch_idx == 1 or self.batch_idx == 100 or self.batch_idx == 1000:
                self.save_pretraining()

            generated_rewards = torch.zeros_like(greedy_rewards)

            summs, targets = self.warmup_oful(valid_sentences, scorers)

            predicted_scores = self.model.pretraining_output(
                sent_contents, summs
            ).squeeze()

            loss = (targets.to(valid_sentences.device) - predicted_scores) ** 2
            loss = loss.sum() / batch_size

            return generated_rewards, loss, greedy_rewards
        else:
            return greedy_rewards

    def save_pretraining(self):
        save_path = f"{self.pretraining_path}/{self.batch_idx}_batches.pt"

        filtered_model_dict = {
            k: v for k, v in self.state_dict().items() if not "decoder" in k
        }

        torch.save(filtered_model_dict, save_path)

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

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
            "test_greedy_rouge_1": greedy_rewards[:, 0],
            "test_greedy_rouge_2": greedy_rewards[:, 1],
            "test_greedy_rouge_L": greedy_rewards[:, 2],
            "test_greedy_rouge_mean": greedy_rewards.mean(-1),
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
                {"params": self.model.pretraining_decoder.parameters(),},
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
            num_workers=6,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        dataset = self.splits["val"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.fields, self.reward_builder, subset="val"),
            batch_size=self.test_batch_size,
            num_workers=6,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        dataset = self.splits["test"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(
                self.fields, self.reward_builder, subset="test"
            ),
            batch_size=self.test_batch_size,
            num_workers=6,
            pin_memory=True,
            drop_last=True,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return LinSIT(dataset, reward, config,)


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

    def get_sents_from_summs(self, sent_contents, sampled_summs):
        all_sents = []

        for sents_doc, sampled_sents in zip(sent_contents, sampled_summs):
            for summ in sampled_sents:
                all_sents.append(
                    torch.stack([sents_doc[sent_id] for sent_id in summ]).sum(0)
                )

        return torch.stack(all_sents)

    def pretraining_output(self, sent_contents, sampled_summs):
        summ_contents = self.get_sents_from_summs(sent_contents, sampled_summs)
        predicted_scores = self.pretraining_decoder(summ_contents)

        return predicted_scores

    def forward(self, x):
        pass
