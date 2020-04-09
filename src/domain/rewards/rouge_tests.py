from src.domain.dataset import CnnDailyMailDataset
from src.domain.utils import set_random_seed, configure_logging, datetime_tqdm
from src.domain.rewards.rouge_python import RougeReward
from src.domain.rewards.rouge import RougeRewardScorer

from graal_utils import timed

import logging
import numpy as np
import json
import os
import pickle
from joblib import delayed, Parallel


@delayed
def python_rouge_sample(fpath, article_summs):
    rouge_reward = RougeReward(n_jobs=1)

    with open(fpath, "rb") as f:
        article = json.load(f)

    summs = []
    if len(article["article"]) > 6:
        for article_summ in article_summs:
            hyp_summ = [article["article"][i] for i in article_summ]
            summs.append(rouge_reward([hyp_summ], [article["abstract"]], "cpu"))

        return summs


@timed
def python_rouge(fpaths, samples):
    summ_scores = Parallel(n_jobs=-1)(
        python_rouge_sample(fpath, article_summs)
        for fpath, article_summs in datetime_tqdm(list(zip(fpaths, samples)))
    )

    summ_scores = filter(None, summ_scores)

    return np.asarray([[t.numpy()[0][0] for t in summ] for summ in summ_scores])


@delayed
def numpy_rouge_sample(fpath, article_summs, subset, rouge_npy_path):
    with open(fpath, "rb") as f:
        article = json.load(f)

    summs = []
    if len(article["article"]) > 6:
        scorer = RougeRewardScorer(
            os.path.join(rouge_npy_path, subset, f'{article["id"]}.npy')
        )

        for article_summ in article_summs:
            summs.append(scorer.get_score(article_summ))

    return summs


@timed
def numpy_rouge(fpaths, samples, subset, rouge_npy_path):
    summ_scores = Parallel(n_jobs=-1)(
        numpy_rouge_sample(fpath, article_summs, subset, rouge_npy_path)
        for fpath, article_summs in datetime_tqdm(list(zip(fpaths, samples)))
    )

    summ_scores = list(filter(None, summ_scores))

    return np.asarray(summ_scores)


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

    bad_files = []

    for subset in subsets:
        fpaths = dataset.fpaths[subset]

        samples = [[list(range(3)), list(range(3, 6))] for _ in fpaths]

        python_rouge_scores = python_rouge(fpaths, samples)
        numpy_rouge_scores = numpy_rouge(
            fpaths,
            samples,
            subset=subset,
            rouge_npy_path="/scratch/magod/summarization_datasets/cnn_dailymail/data/rouge_npy/",
        )

        diffs = np.absolute(python_rouge_scores - numpy_rouge_scores).sum(-1).sum(-1)
        outliers = diffs > 0.05
        bad_files.extend([fpaths[i] for i in np.argwhere(outliers).T.tolist()[0]])
        print(outliers.sum())
        print(diffs[outliers].mean())
        print(diffs.max())

    with open("bad_files.pck", "wb") as f:
        pickle.dump(bad_files, f)
