from src.domain.utils import configure_logging, set_random_seed, datetime_tqdm
from src.domain.dataset import CnnDailyMailDataset
from src.domain.rewards.rouge_python import RougeReward

import logging
import os
import argparse
import json
import numpy as np
import pickle

from itertools import combinations, permutations

SLICE_SIZE = 1000


def main(options):
    configure_logging()

    fpaths = CnnDailyMailDataset(
        "/scratch/magod/summarization_datasets/cnn_dailymail/data/",
        "glove.6B.100d",
        dev=False,
        vectors_cache="/scratch/magod/embeddings/",
        sets=[options.dataset],
    ).fpaths[options.dataset][options.run_index * SLICE_SIZE : (options.run_index + 1) * SLICE_SIZE]

    reward = RougeReward(n_jobs=-1)
    save_dir = os.path.join(options.target_dir, options.dataset)

    for fpath in datetime_tqdm(fpaths, desc="Calculating rouge scores"):
        with open(fpath, "rb") as f:
            article = json.load(f)
        all_summ_idxs = list(combinations(range(len(article["article"])), 3))
        all_summs = [[article["article"][i] for i in idxs] for idxs in all_summ_idxs]
        all_refs = [article["abstract"] for _ in all_summs]

        rouge_scores = reward(all_summs, all_refs, "cpu").squeeze().tolist()
        n_sents = len(article["article"])

        matrix_data = np.zeros((n_sents, n_sents, n_sents, 3), dtype=np.float32)
        for idxs, rouge in zip(all_summ_idxs, rouge_scores):

            for i in permutations(idxs):
                matrix_data[tuple(i)] = np.asarray(rouge)

        np.save(os.path.join(save_dir, f'{article["id"]}.npy'), matrix_data)

    logging.info("Done")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("run_index", type=int)
    argument_parser.add_argument("--data_path", type=str, default="./data/cnn_dailymail")
    argument_parser.add_argument("--dataset", type=str, default="train")
    argument_parser.add_argument("--vectors_cache", type=str, default="./data/embeddings")
    argument_parser.add_argument("--target_dir", type=str, default="./data/cnn_dailymail/rouge_npy/")
    options = argument_parser.parse_args()
    main(options)
