from src.domain.loader_utils import TextDataCollator, NGRAMSLoader
from src.domain.linsit import LinSITExpProcess

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
from tqdm import tqdm
import torch.multiprocessing as mp


class LinSITExp:
    def __init__(self, dataset, reward, hparams):
        super(LinSITExp, self).__init__()
        self.log_path = hparams.log_path
        self.n_mcts_samples = hparams.n_mcts_samples
        self.pad_idx = dataset.pad_idx

        os.makedirs(self.log_path, exist_ok=True)

        if hparams.n_jobs_for_mcts == -1:
            self.n_processes = os.cpu_count()
        else:
            self.n_processes = hparams.n_jobs_for_mcts
        self.pool = mp.Pool(processes=self.n_processes)

        self.dataloader = DataLoader(
            dataset.get_splits()["train"],
            collate_fn=TextDataCollator(
                dataset.fields,
                reward,
                subset="train",
                n_grams_loader=NGRAMSLoader(hparams.data_path),
            ),
            batch_size=hparams.test_batch_size,
            num_workers=2,
            drop_last=False,
        )

    def linsit_exp(
        self, sent_contents, priors, scorers, ids, c_pucts,
    ):
        results = self.pool.map(
            LinSITExpProcess(n_samples=self.n_mcts_samples, c_pucts=c_pucts,),
            zip(sent_contents, priors, scorers, ids),
        )

        return [r for res in results for r in res]

    def process_all(self):
        for batch_idx, batch in enumerate(tqdm(self.dataloader)):
            (_, contents, _, _, ids, scorers, n_grams_dense,) = batch

            n_grams_dense = [torch.from_numpy(_) for _ in n_grams_dense]

            torch.set_grad_enabled(False)

            c_pucts = [10 ** 5, 10 ** 4]

            valid_tokens = ~(contents == self.pad_idx)
            sentences_len = valid_tokens.sum(dim=-1)
            valid_sentences = sentences_len > 0

            priors = torch.ones_like(valid_sentences, dtype=torch.float32)
            priors /= valid_sentences.sum(-1, keepdim=True)

            results = self.linsit_exp(n_grams_dense, priors, scorers, ids, c_pucts,)

            keys = [r[0] for r in results]
            theta_hat_predictions = [r[1] for r in results]

            self.save_results(keys, theta_hat_predictions, batch_idx)

        self.join_all_results()

    def save_results(self, keys, theta_hat_predictions, batch_idx):
        d = {}

        for key, preds in zip(keys, theta_hat_predictions):
            d[key] = preds

        with open(os.path.join(self.log_path, f"results_{batch_idx}.pck"), "wb") as f:
            pickle.dump(d, f)

    def join_all_results(self):
        all_dict_paths = os.listdir(self.log_path)
        d = {}

        for path in all_dict_paths:
            with open(os.path.join(self.log_path, path), "rb") as f:
                d_i = pickle.load(f)

            for k, v in d_i.items():
                d[k] = v

        with open(os.path.join(self.log_path, "results.pck"), "wb") as f:
            pickle.dump(d, f)

    @staticmethod
    def from_config(dataset, reward, config):
        return LinSITExp(dataset, reward, config,)
