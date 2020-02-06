from src.domain.rewards.rouge import RougeReward

from rouge.rouge import Rouge

from graal_utils import timed

import torch


def pltrdy_rouge_func(scorer, hyps, refs):
    hyps = [".".join(hyp) for hyp in hyps]
    refs = [".".join(ref) for ref in refs]
    batch_scores = scorer.get_scores(hyps, refs)

    return torch.tensor(
        [
            (
                batch_score["rouge-1"]["f"],
                batch_score["rouge-2"]["f"],
                batch_score["rouge-l"]["f"],
            )
            for batch_score in batch_scores
        ]
    )


@timed
def pltrdy_rouge(dataset):
    scorer = Rouge()
    score_func = lambda hyps, refs: pltrdy_rouge_func(scorer, hyps, refs)

    model = Lead3(dataset, reward=score_func)
    model.test()


@timed
def pythonrouge_multithread(dataset):
    model = Lead3(dataset, reward=RougeReward())
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
        "./data/cnn_dailymail",
        "glove.6B.100d",
        dev=False,
        sets=["test"],
        max_sents_per_article=None,
        max_tokens_per_sent=None,
    )

    pltrdy_rouge(dataset)
    pythonrouge_multithread(dataset)
