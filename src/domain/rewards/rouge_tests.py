from src.domain.dataset import CnnDailyMailDataset
from src.domain.utils import set_random_seed, configure_logging
from src.domain.loader_utils import TextDataCollator
from src.domain.rewards.rouge import RougeRewardBuilder
from src.domain.rewards.rouge_pearl import RougePearlReward
from src.domain.rewards.rouge_python import RougePythonReward

import logging
import numpy as np
import json
import os
import pickle
from tqdm import tqdm

from rouge import Rouge

rouge = Rouge()


def RougeTest_rouge(ref, hyp, max_num_of_bytes=-1):
    ref = [_.lower() for _ in ref]
    ref = [
        " ".join(ref)
    ]  # join for managing the cases where we have different number of sentence.
    hyp = [_.lower().replace(".", " .") for _ in hyp]
    hyp = [" ".join(hyp)]

    if max_num_of_bytes > 0:
        ref = cutwords(ref)
        hyp = cutwords(hyp)

    rouge_score = rouge.get_scores(hyp, ref)
    return (
        rouge_score[0]["rouge-1"]["f"],
        rouge_score[0]["rouge-2"]["f"],
        rouge_score[0]["rouge-l"]["f"],
    )


def cutwords(sens, max_num_of_chars):
    output = []
    quota = max_num_of_chars
    for sen in sens:
        if quota > len(sen):
            output.append(sen)
            quota -= len(sen)
        else:
            output.append(sen[:quota])
            break
    return output


if __name__ == "__main__":
    configure_logging()
    logging.info("Begin")
    set_random_seed(42)
    subsets = ["val"]

    dataset = CnnDailyMailDataset(
        "./data/cnn_dailymail",
        "glove.6B.100d",
        dev=False,
        vectors_cache="./data/embeddings/",
        sets=subsets,
    )
    dataset = dataset.get_splits()["val"]
    collator = TextDataCollator(
        RougeRewardBuilder("./data/cnn_dailymail/rouge_npy/"), "val"
    )

    samples = [[list(range(3)), list(range(3, 6))] for _ in dataset]
    rouge_python = RougePythonReward()
    rouge_pearl = RougePearlReward()

    for data in tqdm(dataset):
        batch = collator([data])

        raw_contents, contents, raw_abstracts, abstracts, ids, scorers = batch.values()

        l3 = raw_contents[0][:3]

        scores_python = rouge_python([l3], raw_abstracts)
        # scores_pearl = rouge_pearl(
        #     [[" ".join(l3)]], [[[" ".join(r)] for r in raw_abstracts]]
        # )
        scores_numpy = scorers[0].get_score([0, 1, 2])

        diff = np.abs(scores_numpy - scores_python).mean()
        if diff > 0.04:
            print(diff, scores_python, scores_numpy, ids)

