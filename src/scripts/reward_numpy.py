from src.domain.utils import configure_logging, datetime_tqdm
from src.domain.dataset import CnnDailyMailDataset

import logging
import os
import argparse
import json
import numpy as np

from joblib import Parallel, delayed
from collections import OrderedDict
from math import ceil


@delayed
def process_sample(fname, saving_dir, read_dir):
    with open(os.path.join(read_dir, fname), "rb") as f:
        data = json.load(f)

    n_sents = ceil(len(data) ** (1 / 3)) + 1
    assert n_sents * (n_sents - 1) * (n_sents - 2) == len(data)
    matrix_data = np.zeros((n_sents, n_sents, n_sents, 3), dtype=np.float32)
    for key, rouge in data.items():
        idx = np.asarray(key.split('-'), dtype=int)
        matrix_data[tuple(idx)] = np.asarray(rouge)

    f_id = fname.split(".")[0]
    np.save(os.path.join(saving_dir, f_id), matrix_data)


def main(options):
    configure_logging()

    saving_dir = os.path.join(options.target_dir, options.dataset)
    os.makedirs(saving_dir, exist_ok=True)

    read_dir = os.path.join(options.source_dir, options.dataset)

    Parallel(n_jobs=-1)(
        process_sample(fname, saving_dir, read_dir)
        for fname in datetime_tqdm(os.listdir(read_dir), desc="Saving rouge scores")
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
    argument_parser.add_argument(
        "--source_dir", type=str, default="./data/cnn_dailymail/rouge/"
    )
    options = argument_parser.parse_args()
    main(options)
