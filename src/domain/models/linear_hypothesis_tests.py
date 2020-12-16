from src.domain.loader_utils import TextDataCollator

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import combinations, product


class LinearHypothesisTests(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        super(LinearHypothesisTests, self).__init__()
        self.reward_builder = reward
        self.pad_idx = dataset.pad_idx
        self.fields = dataset.fields

        self.embedding_dim = dataset.embedding_dim
        self.splits = dataset.get_splits()
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

        self.__build_model(dataset)
        self.model = RLSummModel(hparams.hidden_dim, hparams.decoder_dim, self.dropout,)

        self.results = []
        self.config_args = {
            "normalize": [True],
            "use_torchtext": [True],
            "n": [2],
            "add_bias": [False],
            "pca_dim": [50],
            "unif_norm": [True],
        }

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

    def forward(self, batch, subset):
        raw_contents, contents, raw_abstracts, abstracts, ids, scorers = batch
        batch_size = len(contents)

        for doc_contents, raw_doc_contents, doc_scorer in zip(
            contents.tolist(), raw_contents, scorers
        ):
            for config in list(product(*[v for k, v in self.config_args.items()])):
                normalize, use_torchtext, n, add_bias, pca_dim, unif_nom = config

                try:
                    # Get n-grams
                    ngrams = get_ngrams_dense(
                        doc_contents,
                        self.pad_idx,
                        normalize=normalize,
                        n=n,
                        add_bias=add_bias,
                        pca_dim=pca_dim,
                        unif_norm=unif_nom,
                    )

                    # Get lstsq results with specific config
                    results = run_lstsq(ngrams, doc_scorer.scores)

                    # save results
                    self.results.append(results)
                except Exception as e:
                    print(config)
                    raise e

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
        self.forward(batch, subset="test")

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
        theta_norms = np.array([r[0] for r in self.results])
        mean_res = np.array([float(r[1]) for r in self.results if len(r[1]) > 0])

        print("TEST RESULTS")
        print(
            f"Percen times theta norm < 1 : {(theta_norms <= 1).sum() / len(self.results) * 100} %s"
        )
        print(f"Mean residual : {np.mean(mean_res)} ± {np.std(mean_res)}")

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
            num_workers=4,
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
            num_workers=4,
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
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return LinearHypothesisTests(dataset, reward, config,)


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


def run_lstsq(ngrams, scores):
    n_sents = len(ngrams)
    all_summs = [list(c) for c in combinations(range(n_sents), 3)]
    all_summs_reps = np.stack([ngrams[summ].sum(0) for summ in all_summs])

    scores = scores.mean(-1)
    all_summs_scores = np.stack([scores[tuple(summ)] for summ in all_summs])

    theta, res, rank, s = np.linalg.lstsq(all_summs_reps, all_summs_scores, rcond=-1)

    return np.linalg.norm(theta), res / len(all_summs)

