from src.domain.utils import configure_logging, set_random_seed, datetime_tqdm
from src.domain.dataset import CnnDailyMailDataset
from src.domain.rewards.rouge_python import RougeReward

import logging
import os
import argparse
import json

from itertools import permutations

SLICE_SIZE = 300


def main(options):
    configure_logging()

    begin_idx = options.slice * SLICE_SIZE
    end_idx = begin_idx + SLICE_SIZE

    logging.info(f"Beginning rouge script for slice {options.slice}")

    dataset = CnnDailyMailDataset(
        "./data/cnn_dailymail",
        "glove.6B.100d",
        sets=["train"],
        begin_idx=begin_idx,
        end_idx=end_idx,
        dev=False,
    )
    reward = RougeReward(n_jobs=-1)

    iterable = list(zip(dataset.fpaths["train"], dataset.get_splits()["train"],))

    for fpath, article in datetime_tqdm(iterable, desc="Calculating rouge scores"):
        all_summ_idxs = list(permutations(range(len(article.raw_content)), 3))
        all_summs = [[article.raw_content[i] for i in idxs] for idxs in all_summ_idxs]
        all_refs = [article.raw_abstract for _ in all_summs]

        rouge_scores = reward(all_summs, all_refs, "cpu").squeeze().tolist()
        rouge_scores_dict = {
            "-".join([str(i) for i in summ]): rouge_score
            for summ, rouge_score in zip(all_summ_idxs, rouge_scores)
        }

        with open(fpath, "rb") as f:
            data = json.load(f)

        data.update({"rouge": rouge_scores_dict})

        with open(fpath, "w", encoding="utf8") as f:
            json.dump(data, f)

        logging.info(f'Done file {fpath}')

    logging.info("Done")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("slice", type=int)
    options = argument_parser.parse_args()
    main(options)
