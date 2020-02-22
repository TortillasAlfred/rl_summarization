from src.domain.rewards.rouge import RougeReward

from graal_utils import timed

import torch

import rouge


def diego_rouge_func(evaluator, hyps, refs):
    hyps = [' . '.join(h) for h in hyps]
    refs = [[' . '.join(r)] for r in refs]
    results = evaluator.get_scores(hyps, refs)
    results = [[pair['f'][0] for pair in results[metric]]
               for metric in ['rouge-1', 'rouge-2', 'rouge-l']]
    results = [[r[i] for r in results] for i in range(len(results[0]))]
    return torch.FloatTensor(results)


@timed
def diego_rouge_no_stemming(dataset):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=False,
                            apply_avg=False,
                            stemming=False,
                            ensure_compatibility=False)

    def score_func(hyps, refs, device): return diego_rouge_func(
        evaluator, hyps, refs)
    model = Lead3(dataset, score_func)
    model.test()


@timed
def diego_rouge_stemming(dataset):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=False,
                            apply_avg=False,
                            stemming=True,
                            ensure_compatibility=False)
    rouge.Rouge.load_stemmer(ensure_compatibility=True)
    def score_func(hyps, refs, device): return diego_rouge_func(
        evaluator, hyps, refs)
    model = Lead3(dataset, score_func)
    model.test()


@timed
def pythonrouge_multithread_stemming(dataset):
    model = Lead3(dataset, reward=RougeReward())
    model.test()


@timed
def pythonrouge_multithread_no_stemming(dataset):
    model = Lead3(dataset, reward=RougeReward(stemming=False))
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
        sets=["test"]
    )

    diego_rouge_no_stemming(dataset)
    pythonrouge_multithread_no_stemming(dataset)
    diego_rouge_stemming(dataset)
    pythonrouge_multithread_stemming(dataset)
