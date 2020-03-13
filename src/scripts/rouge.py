from src.domain.utils import configure_logging, set_random_seed, datetime_tqdm
from src.domain.dataset import CnnDailyMailDataset
from src.domain.rewards.rouge_python import RougeReward

import logging
import os
import argparse
import json
import numpy as np

from itertools import combinations, permutations

SLICE_SIZE = 400


def main(options):
    configure_logging()

    reward = RougeReward(n_jobs=-1)

    for dataset in ["test", "val", "train"]:
        save_dir = os.path.join(options.target_dir, dataset)

        os.makedirs(save_dir, exist_ok=True)

        bad_files_dir = "/scratch/magod/summarization_datasets/bad_files/"

        with open(os.path.join(bad_files_dir, f"{dataset}.pck"), "rb") as f:
            iterable = pickle.load(f)

        for fpath in datetime_tqdm(iterable, desc="Calculating rouge scores"):
            article = json.load(fpath)
            all_summ_idxs = list(combinations(range(len(article["article"])), 3))
            all_summs = [
                [article["article"][i] for i in idxs] for idxs in all_summ_idxs
            ]
            all_refs = [article["abstract"] for _ in all_summs]

            rouge_scores = reward(all_summs, all_refs, "cpu").squeeze().tolist()
            n_sents = len(article["article"])

            matrix_data = np.zeros((n_sents, n_sents, n_sents, 3), dtype=np.float32)
            for idxs, rouge in zip(all_summ_idxs, rouge_scores):

                for i in permutations(idxs):
                    matrix_data[tuple(idxs)] = np.asarray(rouge)

            fname = fpath.split("/")[-1].split(".")[0]
            np.save(os.path.join(save_dir, fname), matrix_data)

    logging.info("Done")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--data_path", type=str, default="./data/cnn_dailymail"
    )
    argument_parser.add_argument("--dataset", type=str, default="train")
    argument_parser.add_argument(
        "--vectors_cache", type=str, default="./data/embeddings"
    )
    argument_parser.add_argument(
        "--target_dir", type=str, default="./data/cnn_dailymail/rouge_npy/"
    )
    options = argument_parser.parse_args()
    main(options)
