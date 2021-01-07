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
from torch.utils.data.dataloader import DataLoader

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
    subsets = ["test"]

    dataset = CnnDailyMailDataset(
        "./data/cnn_dailymail",
        "glove.6B.100d",
        dev=False,
        vectors_cache="./data/embeddings/",
        sets=subsets,
    )
    subset = dataset.get_splits()["test"]
    collator = TextDataCollator(dataset.fields, None, "test",)
    rouge_python = RougePythonReward()
    scores_python = []

    loader = DataLoader(
        subset, collate_fn=collator, batch_size=64, num_workers=0, drop_last=False,
    )

    for batch in tqdm(loader):
        raw_contents, contents, raw_abstracts, abstracts, ids = batch

        scores_python.append(
            rouge_python.get_scores(
                [list(range(3)) for _ in range(len(ids))], raw_contents, raw_abstracts
            )
        )
        # scores_pearl = rouge_pearl(
        #     [[" ".join(l3)]], [[[" ".join(r)] for r in raw_abstracts]]
        # )
        # scores_numpy = scorers[0]([0, 1, 2])

        # diff = np.abs(scores_numpy - scores_python).mean()
        # if diff > 0.04:
        #     print(diff, scores_python, scores_numpy, ids)

    print(np.vstack(scores_python).mean(0))

