import numpy as np
import os


class RougeRewardBuilder:
    def __init__(self, base_path):
        self.base_path = base_path

    def init_scorer(self, article_id, subset):
        assert subset in ["train", "test", "val"]

        read_path = os.path.join(self.base_path, subset, f"{article_id}.npy")

        return RougeRewardScorer(read_path)

    @staticmethod
    def from_config(config):
        return RougeRewardBuilder(config.rouge_base_path)


class RougeRewardScorer:
    def __init__(self, read_path):
        self.scores = np.load(read_path)

    def get_score(self, state):
        summ_ids = state.summary_idxs
        if len(summ_ids) == 3:
            return self.scores[tuple(sorted(summ_ids))]
        else:
            return [0.0, 0.0, 0.0]
