from src.domain.dataset import CnnDailyMailDataset
from src.domain.utils import set_random_seed, configure_logging, datetime_tqdm
from src.domain.rewards.rouge_python import RougeReward
from src.domain.rewards.rouge import RougeRewardScorer

from graal_utils import timed

import logging
import numpy as np
import json
import os


@timed
def python_rouge(fpaths, samples):
    rouge_reward = RougeReward(n_jobs=-1)
    summ_scores = []

    for fpath, article_summs in datetime_tqdm(list(zip(fpaths, samples))):
        with open(fpath, "rb") as f:
            article = json.load(f)

        if len(article["article"]) > 5:
            for article_summ in article_summs:
                hyp_summ = [article["article"][i] for i in article_summ]
                summ_scores.append(rouge_reward([hyp_summ], [article["abstract"]], "cpu"))

    return np.asarray([t.numpy()[0][0] for t in summ_scores])


@timed
def numpy_rouge(fpaths, samples, subset, rouge_npy_path):
    summ_scores = []

    for fpath, article_summs in datetime_tqdm(list(zip(fpaths, samples))):
        with open(fpath, "rb") as f:
            article = json.load(f)

        if len(article["article"]) > 5:
            scorer = RougeRewardScorer(
                os.path.join(rouge_npy_path, subset, f'{article["id"]}.npy')
            )
            for article_summ in article_summs:
                summ_scores.append(scorer.get_score(article_summ))

    return np.asarray(summ_scores)


if __name__ == "__main__":
    configure_logging()
    logging.info("Begin")
    set_random_seed(42)

    dataset = CnnDailyMailDataset(
        "/scratch/magod/summarization_datasets/cnn_dailymail/data/",
        "glove.6B.100d",
        dev=False,
        vectors_cache="/scratch/magod/embeddings/",
        sets=["val", "test"],
    )

    for subset in ["val", "test"]:
        fpaths = dataset.fpaths[subset]

        samples = [[list(range(3))] * 1 for _ in fpaths]

        python_rouge_scores = python_rouge(fpaths, samples)
        numpy_rouge_scores = numpy_rouge(
            fpaths,
            samples,
            subset=subset,
            rouge_npy_path="/scratch/magod/summarization_datasets/cnn_dailymail/data/rouge_npy/",
        )

        diffs = np.absolute(python_rouge_scores - numpy_rouge_scores).sum(-1)
        print((diffs > 0).sum())
        print(diffs.mean())
        print(diffs.max())
