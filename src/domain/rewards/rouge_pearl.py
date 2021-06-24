from pythonrouge.pythonrouge import Pythonrouge

from joblib import Parallel, delayed
import numpy as np


@delayed
def rouge_reward(seqs, stemming):
    hyp, ref = seqs
    rouge = Pythonrouge(
        summary_file_exist=False,
        delete_xml=True,
        summary=[hyp],
        reference=[ref],
        n_gram=2,
        ROUGE_SU4=False,
        ROUGE_L=True,
        f_measure_only=True,
        stemming=stemming,
        stopwords=False,
        word_level=True,
        length_limit=False,
        resampling=False,
    )
    score = rouge.calc_score()

    return (score["ROUGE-1"], score["ROUGE-2"], score["ROUGE-L"])


class RougePearlReward:
    def __init__(self, n_jobs=1, stemming=True):
        self.n_jobs = n_jobs
        self.stemming = stemming

    def __call__(self, hyps, refs):
        refs = [[r] for r in refs]

        scores = list(Parallel(n_jobs=self.n_jobs)(rouge_reward(seqs, self.stemming) for seqs in zip(hyps, refs)))

        return np.asarray(scores[0], dtype=np.float32)

    def get_score(self, state):
        if len(state.summary_idxs) == 3:
            hyps = [[state.raw_content[i] for i in state.summary_idxs]]
            refs = [state.raw_abstract]

            return self.__call__(hyps, refs)
        else:
            return np.zeros((3,), dtype=np.float32)

    @staticmethod
    def from_config(self, config):
        return RougePearlReward(n_jobs=config.rouge_jobs)


if __name__ == "__main__":
    from src.domain.dataset import CnnDailyMailDataset
    from src.domain.utils import configure_logging
    import logging

    configure_logging()

    logging.info("Begin")

    dataset = CnnDailyMailDataset("./data/cnn_dailymail", "glove.6B.50d", dev=False, sets=["test"])

    test_loader = dataset.get_loaders(batch_size=64, device="cuda:0")["test"]
    scorer = RougeReward()

    for batch in test_loader:
        (raw_contents, _), (raw_abstracts, _) = batch
        l3 = [r_c[:3] for r_c in raw_contents]

        logging.info(scorer(l3, raw_abstracts))
