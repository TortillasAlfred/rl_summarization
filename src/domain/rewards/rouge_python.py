import rouge
import torch
from torchtext.data import BucketIterator


class RougeReward:
    def __init__(self):
        rouge.Rouge.load_stemmer(ensure_compatibility=True)
        self.evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                     max_n=2,
                                     limit_length=False,
                                     apply_avg=False,
                                     stemming=True,
                                     ensure_compatibility=False)

    def __call__(self, hyps, refs, device):
        hyps = [' . '.join(h) for h in hyps]
        refs = [[' . '.join(r)] for r in refs]
        results = self.evaluator.get_scores(hyps, refs)
        results = [[pair['f'][0] for pair in results[metric]]
                   for metric in ['rouge-1', 'rouge-2', 'rouge-l']]
        results = [[r[i] for r in results] for i in range(len(results[0]))]
        return torch.FloatTensor(results).to(device)

    @staticmethod
    def from_config(self, config):
        return RougeReward()


if __name__ == "__main__":
    from src.domain.dataset import CnnDailyMailDataset
    from src.domain.utils import configure_logging
    import logging

    configure_logging()

    logging.info("Begin")

    dataset = CnnDailyMailDataset(
        "./data/cnn_dailymail", "glove.6B.50d", dev=False, sets=["test"]
    )

    test_split = dataset.get_splits()['test']
    test_loader = BucketIterator(
        test_split,
        train=False,
        batch_size=64,
        shuffle=False,
        sort=False,
        device='cpu',
    )
    scorer = RougeReward()

    for batch in test_loader:
        (raw_contents, _), (raw_abstracts, _) = batch
        l3 = [r_c[:3] for r_c in raw_contents]

        logging.info(scorer(l3, raw_abstracts, 'cpu'))
