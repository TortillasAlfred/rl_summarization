from src.domain.dataset import CnnDailyMailDataset
from src.domain.utils import set_random_seed, configure_logging, datetime_tqdm
from src.domain.rewards.rouge import RougeRewardScorer

from graal_utils import timed

import logging
import numpy as np
import json
import os
import pickle
from joblib import delayed, Parallel


@delayed
def get_outlier(fpath, rouge_npy_path, subset):
    with open(fpath, "rb") as f:
        article = json.load(f)

    article_len = len(article["article"])

    if article_len > 3:
        scores = RougeRewardScorer(
            os.path.join(rouge_npy_path, subset, f'{article["id"]}.npy')
        ).scores
        if scores.shape[0] != article_len:
            return fpath

    return None


if __name__ == "__main__":
    configure_logging()
    logging.info("Begin")
    set_random_seed(42)

    subsets = ["train", "val", "test"]

    dataset = CnnDailyMailDataset(
        "/scratch/magod/summarization_datasets/cnn_dailymail/data/",
        "glove.6B.100d",
        dev=False,
        vectors_cache="/scratch/magod/embeddings/",
        sets=subsets,
    )

    for subset in subsets:
        fpaths = dataset.fpaths[subset]

        bad_formatted_files = Parallel(n_jobs=-1)(
            get_outlier(
                fpath,
                "/scratch/magod/summarization_datasets/cnn_dailymail/data/rouge_npy/",
                subset,
            )
            for fpath in datetime_tqdm(fpaths)
        )

        bad_formatted_files = list(filter(None, bad_formatted_files))
        with open(f"bad_files_{subset}.pck", "wb") as f:
            pickle.dump(bad_formatted_files, f)

        logging.info(
            f"{len(bad_formatted_files)} bad files found for subset '{subset}'"
        )
