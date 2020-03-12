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

    begin_idx = options.slice * SLICE_SIZE
    end_idx = begin_idx + SLICE_SIZE

    logging.info(f"Beginning rouge script for slice {options.slice}")

    dataset = CnnDailyMailDataset(
        options.data_path,
        "glove.6B.100d",
        vectors_cache=options.vectors_cache,
        sets=[options.dataset],
        begin_idx=begin_idx,
        end_idx=end_idx,
        dev=False,
    )
    reward = RougeReward(n_jobs=-1)

    iterable = list(
        zip(dataset.fpaths[options.dataset], dataset.get_splits()[options.dataset],)
    )

    article_lens = [len(art.raw_content) for _, art in iterable]
    logging.info(f"Article lengths are : {article_lens}")
    save_dir = os.path.join(options.target_dir, options.dataset)

    os.makedirs(save_dir, exist_ok=True)

    for fpath, article in datetime_tqdm(iterable, desc="Calculating rouge scores"):
        all_summ_idxs = list(combinations(range(len(article.raw_content)), 3))
        all_summs = [[article.raw_content[i] for i in idxs] for idxs in all_summ_idxs]
        all_refs = [article.raw_abstract for _ in all_summs]

        rouge_scores = reward(all_summs, all_refs, "cpu").squeeze().tolist()
        n_sents = len(article.raw_content)

        matrix_data = np.zeros((n_sents, n_sents, n_sents, 3), dtype=np.float32)
        for idxs, rouge in zip(all_summ_idxs, rouge_scores):

            for i in permutations(idxs):
                matrix_data[tuple(idxs)] = np.asarray(rouge)

        fname = fpath.split("/")[-1].split(".")[0]
        np.save(os.path.join(save_dir, fname), matrix_data)

        logging.info(
            f"Done file {fpath} that contained {len(article.raw_content)} sentences."
        )

    logging.info("Done")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("slice", type=int)
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
