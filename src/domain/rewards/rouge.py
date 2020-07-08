import numpy as np
import os


class RougeRewardBuilder:
    def __init__(self, base_path):
        self.base_path = base_path

    def init_scorer(self, article_id, subset):
        assert subset in ["train", "test", "val"]

        if subset == "test":
            subset = "val"

        read_path = os.path.join(self.base_path, subset, f"{article_id}.npy")

        return RougeRewardScorer(read_path)

    @staticmethod
    def from_config(config):
        return RougeRewardBuilder(config.rouge_base_path)


class RougeRewardScorer:
    def __init__(self, read_path):
        self.scores = np.load(read_path)

    def __call__(self, summary_idxs):
        if len(summary_idxs) == 3:
            return self.scores[tuple(sorted(summary_idxs))]
        else:
            return np.zeros((3,), dtype=np.float32)

    def get_score(self, summary_idxs):
        return self.__call__(summary_idxs)
