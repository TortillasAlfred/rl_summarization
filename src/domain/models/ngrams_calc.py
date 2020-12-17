from src.domain.loader_utils import TextDataCollator, NGRAMSSaver

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm


class NGramsPCA:
    def __init__(self, dataset, log_path):
        super(NGramsPCA, self).__init__()
        self.log_path = log_path
        self.n_jobs = 4
        self.ngrams = NGRAMSSaver(self.log_path, "train", dataset.pad_idx)

        for subset in ["train"]:
            os.makedirs(os.path.join(self.log_path, subset), exist_ok=True)

        self.dataloader = DataLoader(
            dataset.get_splits()["train"],
            collate_fn=TextDataCollator(dataset.fields, None, subset="train",),
            batch_size=128,
            num_workers=0,
            drop_last=False,
        )

    def process_all(self):
        for batch in tqdm(self.dataloader):
            (_, contents, _, _, ids,) = batch

            Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self.ngrams)(*doc) for doc in zip(contents, ids)
            )

    @staticmethod
    def from_config(dataset, reward, config):
        return NGramsPCA(dataset, config.log_path,)
