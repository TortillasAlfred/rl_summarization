from src.domain.utils import configure_logging, datetime_tqdm
from src.domain.dataset import CnnDailyMailDataset

import logging
import os
import argparse
import json
import numpy as np
import pickle

from joblib import Parallel, delayed
from collections import OrderedDict
from itertools import permutations
from math import ceil


@delayed
def process_sample(fpath, saving_dir, bad_files_path):
    with open(fpath, "rb") as f:
        data = json.load(f)

    if "rouge" in data:
        rouge_scores = data["rouge"]

        del data["rouge"]

        n_sents = ceil(len(rouge_scores) ** (1 / 3)) + 1
        assert n_sents * (n_sents - 1) * (n_sents - 2) == len(data)
        matrix_data = np.zeros((n_sents, n_sents, n_sents, 3), dtype=np.float32)
        for key, rouge in data.items():
            idx = np.asarray(key.split("-"), dtype=int)

            for i in permutations(idx):
                matrix_data[tuple(idx)] = np.asarray(rouge)

        np.save(os.path.join(saving_dir, data["id"]), matrix_data)

        with open(fpath, "w", encoding="utf8") as f:
            json.dump(data, f)
    else:
        with open(bad_files_path, "rb") as f:
            bad_files = pickle.load(f)

        bad_files.append(fpath)

        with open(bad_files_path, "wb") as f:
            pickle.dump(bad_files, f)


def main(options):
    configure_logging()

    saving_dir = os.path.join(options.target_dir, options.dataset)
    os.makedirs(saving_dir, exist_ok=True)

    bad_files_path = os.path.join(saving_dir, "bad_train_files.pck")

    with open(bad_files_path, "wb") as f:
        pickle.dump([], f)

    dataset = CnnDailyMailDataset(
        options.data_path,
        "glove.6B.100d",
        vectors_cache=options.vectors_cache,
        sets=[options.dataset],
        dev=options.dev,
    )

    Parallel(n_jobs=-1)(
        process_sample(fname, saving_dir, bad_files_path)
        for fname in datetime_tqdm(
            dataset.fpaths[options.dataset], desc="Saving rouge scores"
        )
    )

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
    argument_parser.add_argument("--dev", action="store_true")
    options = argument_parser.parse_args()
    main(options)
