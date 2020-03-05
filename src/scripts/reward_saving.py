from src.domain.utils import configure_logging, datetime_tqdm
from src.domain.dataset import CnnDailyMailDataset

import logging
import os
import argparse
import json

from joblib import Parallel, delayed


@delayed
def process_sample(fpath, saving_dir):
    with open(fpath, "rb") as f:
        data = json.load(f)

    if "rouge" in data:
        rouge_scores = data["rouge"]

        del data["rouge"]

        with open(
            os.path.join(saving_dir, f"{data['id']}.json"), "w", encoding="utf8"
        ) as f:
            json.dump(rouge_scores, f)

        with open(fpath, "w", encoding="utf8") as f:
            json.dump(data, f)
    else:
        logging.info(f"file {data['id']} was found with no rouge dict")


def main(options):
    configure_logging()

    dataset = CnnDailyMailDataset(
        options.data_path,
        "glove.6B.100d",
        vectors_cache=options.vectors_cache,
        sets=[options.dataset],
        dev=False,
    )

    saving_dir = os.path.join(options.target_dir, options.dataset)
    os.makedirs(saving_dir, exist_ok=True)

    Parallel(n_jobs=-1)(
        process_sample(fpath, saving_dir)
        for fpath in datetime_tqdm(
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
        "--target_dir", type=str, default="./data/cnn_dailymail/rouge/"
    )
    options = argument_parser.parse_args()
    main(options)
