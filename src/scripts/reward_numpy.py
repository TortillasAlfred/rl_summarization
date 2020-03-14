from src.domain.utils import configure_logging, datetime_tqdm
from src.domain.dataset import CnnDailyMailDataset

import logging
import os
import argparse
import json
import numpy as np
import pickle

from joblib import Parallel, delayed
from itertools import permutations


@delayed
def process_sample(fpath, saving_dir):
    with open(fpath, "rb") as f:
        data = json.load(f)

    f_id = fpath.split("/")[-1].split(".")[0]
    npy_old_path = os.path.join(saving_dir, f_id) + ".npy"
    npy_new_path = os.path.join(saving_dir, data["id"]) + ".npy"

    os.rename(npy_old_path, npy_new_path)


def main(options):
    configure_logging()

    dataset = CnnDailyMailDataset(
        options.data_path,
        "glove.6B.100d",
        vectors_cache=options.vectors_cache,
        sets=["train", "val", "test"],
        dev=False,
    )

    for dataset in ["test", "val", "train"]:
        saving_dir = os.path.join(options.target_dir, dataset)
        os.makedirs(saving_dir, exist_ok=True)

        Parallel(n_jobs=1)(
            process_sample(fname, saving_dir)
            for fname in datetime_tqdm(
                dataset.fpaths[dataset], desc="Saving rouge scores"
            )
        )

    logging.info("Done")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--data_path", type=str, default="./data/cnn_dailymail"
    )
    argument_parser.add_argument(
        "--vectors_cache", type=str, default="./data/embeddings"
    )
    argument_parser.add_argument(
        "--target_dir", type=str, default="./data/cnn_dailymail/rouge_npy/"
    )
    options = argument_parser.parse_args()
    main(options)
