from src.domain.rewards.rouge import RougeReward

from rouge.rouge import Rouge

from graal_utils import timed

import torch


def preprocess_texts(texts, pad_idx, itos):
    texts = [t.tolist() if isinstance(t, torch.Tensor) else t for t in texts]
    texts = [[[w for w in s if w is not pad_idx] for s in text]
             for text in texts]
    texts = [[s for s in text if len(s) > 0] for text in texts]

    texts = [[[itos[w] for w in s] for s in t] for t in texts]
    texts = [' . '.join([' '.join(s) for s in t]) for t in texts]

    return texts


@timed
def original_rouge(loader, scorer, itos, pad_idx):
    scores = []

    for batch in loader:
        content, abstract = batch
        content = content[:, :3]

        content = preprocess_texts(content, pad_idx, itos=itos)
        abstract = preprocess_texts(abstract, pad_idx, itos=itos)

        batch_scores = scorer.get_scores(content, abstract)
        scores.append(
            (batch_scores[0]['rouge-1']['f'], batch_scores[0]['rouge-2']['f'],
             batch_scores[0]['rouge-l']['f']))

    return scores


@timed
def own_rouge(loader, scorer, pad_idx):
    scores = []

    for batch in loader:
        content, abstract = batch
        content = content[:, :3]

        batch_scores = scorer(content, abstract, pad_idx)
        scores.append(tuple(batch_scores.view(-1).tolist()))

    return scores


if __name__ == '__main__':
    from src.domain.dataset import CnnDailyMailDataset
    from src.domain.utils import set_random_seed
    import logging

    logging.info('Begin')
    set_random_seed(42)

    dataset = CnnDailyMailDataset('./data/cnn_dailymail',
                                  'glove.6B.300d',
                                  dev=True,
                                  sets=['test'],
                                  max_sents_per_article=None,
                                  max_tokens_per_sent=None)
    original_scorer = Rouge()
    own_scorer = RougeReward(itos=dataset.itos,
                             pad_idx=dataset.pad_idx,
                             avg=False)

    test_loader_cpu = dataset.get_loaders(batch_size=1,
                                          device='cpu',
                                          shuffle=False)['test']

    original_scores_cpu = original_rouge(test_loader_cpu, original_scorer,
                                         dataset.itos, dataset.pad_idx)

    own_scores_cpu = own_rouge(test_loader_cpu, own_scorer, dataset.pad_idx)

    print('done')