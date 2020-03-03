from src.domain.rewards.rouge_pearl import RougeReward as RougePearl
from src.domain.rewards.rouge_python import RougeReward as RougePython

from graal_utils import timed

import torch


@timed
def diego_rouge_stemming(dataset, n_jobs):
    model = Lead3(dataset, reward=RougePython(n_jobs))
    model.test()


@timed
def pythonrouge_multithread_stemming(dataset):
    model = Lead3(dataset, reward=RougePearl())
    model.test()


if __name__ == "__main__":
    from src.domain.dataset import CnnDailyMailDataset
    from src.domain.utils import set_random_seed, configure_logging
    from src.domain.models.baselines import Lead3
    import logging

    configure_logging()
    logging.info("Begin")
    set_random_seed(42)

    dataset = CnnDailyMailDataset(
        "./data/cnn_dailymail", "glove.6B.100d", dev=False, sets=["test"]
    )

    diego_rouge_stemming(dataset, 1)
    diego_rouge_stemming(dataset, -1)
    pythonrouge_multithread_stemming(dataset)
