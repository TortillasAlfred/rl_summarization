from src.domain.dataset import CnnDailyMailDataset
from src.domain.utils import set_random_seed, configure_logging, datetime_tqdm

from graal_utils import timed

import logging
import numpy as np
import json
import os


@timed
def json_rouge(dataset, samples):
    json_rouge_dir = "./data/cnn_dailymail/rouge/test/"
    summ_scores = []

    for article, summs in datetime_tqdm(list(zip(dataset, samples))):
        with open(os.path.join(json_rouge_dir, f"{article.id}.json"), "rb") as f:
            scores = json.load(f)

        for summ in summs:
            summ_scores.append(scores["-".join([str(s) for s in summ])])

    summ_scores = np.asarray(summ_scores)

    return summ_scores.mean(0)


@timed
def numpy_rouge(dataset, samples):
    npy_rouge_dir = (
        "/scratch/magod/summarization_datasets/cnn_dailymail/data/rouge_npy/test/"
    )
    summ_scores = []

    for article, summs in datetime_tqdm(list(zip(dataset, samples))):
        scores = np.load(os.path.join(npy_rouge_dir, f"{article.id}.npy"))

        for summ in summs:
            summ_scores.append(scores[tuple(summ)])

    summ_scores = np.asarray(summ_scores)

    return summ_scores.mean(0)


if __name__ == "__main__":
    configure_logging()
    logging.info("Begin")
    set_random_seed(42)

    dataset = CnnDailyMailDataset(
        "/scratch/magod/summarization_datasets/cnn_dailymail/data/",
        "glove.6B.100d",
        dev=False,
        sets=["test"],
    ).get_splits()["test"]

    samples = [[list(range(3))] for article in dataset]

    logging.info(numpy_rouge(dataset, samples))
