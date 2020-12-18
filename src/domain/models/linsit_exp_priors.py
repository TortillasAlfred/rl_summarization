from src.domain.loader_utils import TextDataCollator, NGRAMSLoader
from src.domain.linsit import LinSITExpPriorsProcess

import torch
from torch.utils.data import DataLoader
import numpy as np
from itertools import product
import os
import torch.multiprocessing as mp
import pickle
from tqdm import tqdm


class LinSITExpPriors:
    def __init__(self, dataset, reward, hparams):
        super(LinSITExpPriors, self).__init__()
        self.log_path = hparams.log_path
        self.n_mcts_samples = hparams.n_mcts_samples
        self.pad_idx = dataset.pad_idx
        self.taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.c_pucts = np.logspace(7, 11, 5)

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

    def linsit_exp_priors(
        self, sent_contents, greedy_priors, all_prior_choices, scorers, ids, c_puct, tau
    ):
        results = self.pool.map(
            LinSITExpPriorsProcess(
                n_samples=self.n_mcts_samples, c_puct=c_puct, tau=tau
            ),
            zip(sent_contents, greedy_priors, all_prior_choices, scorers, ids,),
        )

        return [r for res in results for r in res]

    def process_all(self):
        for batch_idx, batch in enumerate(tqdm(self.dataloader)):
            (_, contents, _, _, ids, scorers, n_grams_dense,) = batch
            batch_size = len(contents)

            n_grams_dense = [torch.from_numpy(_) for _ in n_grams_dense]

            torch.set_grad_enabled(False)

            valid_tokens = ~(contents == self.pad_idx)
            sentences_len = valid_tokens.sum(dim=-1)
            valid_sentences = sentences_len > 0

            prior_choices = ["best", "med", "worst"]
            greedy_priors, all_prior_choices = self.sample_greedy_priors(
                batch_size, valid_sentences, prior_choices, scorers
            )

            all_keys = []
            all_theta_hat_predictions = []
            for c_puct in self.c_pucts:
                for tau in self.taus:
                    results = self.linsit_exp_priors(
                        n_grams_dense,
                        greedy_priors,
                        all_prior_choices,
                        scorers,
                        ids,
                        c_puct,
                        tau,
                    )

                    all_keys.extend([res[0] for res in results])
                    all_theta_hat_predictions.extend([res[1] for res in results])

            self.save_results(all_keys, all_theta_hat_predictions, batch_idx)

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

    def sample_greedy_priors(self, batch_size, valid_sentences, prior_choices, scorers):
        greedy_priors = torch.zeros(
            (batch_size, len(prior_choices), valid_sentences.shape[-1]),
            dtype=torch.float32,
            device=valid_sentences.device,
        )
        all_prior_choices = [prior_choices] * batch_size

        for batch_idx, (_, scorer) in enumerate(zip(valid_sentences, scorers)):
            for sample_idx, prior_choice in enumerate(prior_choices):
                s = scorer.scores
                if prior_choice == "best":
                    selected_sents = s.argmax()
                elif prior_choice == "worst":
                    s_pos = np.ma.masked_less_equal(s, 0)
                    selected_sents = s_pos.argmin()
                else:
                    # Get median
                    selected_sents = np.argsort(s)[len(s) // 2]
                selected_sents = torch.from_numpy(
                    scorer.summary_from_idx(selected_sents)
                )
                greedy_priors[batch_idx][sample_idx][selected_sents] = 1 / 3

        return greedy_priors, all_prior_choices

    @staticmethod
    def from_config(dataset, reward, config):
        return LinSITExpPriors(dataset, reward, config,)
