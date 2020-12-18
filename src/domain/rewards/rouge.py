import numpy as np
import os
import tarfile
import logging
from scipy.special import comb

LEN_TO_N = {
    1: 3,
    4: 4,
    10: 5,
    20: 6,
    35: 7,
    56: 8,
    84: 9,
    120: 10,
    165: 11,
    220: 12,
    286: 13,
    364: 14,
    455: 15,
    560: 16,
    680: 17,
    816: 18,
    969: 19,
    1140: 20,
    1330: 21,
    1540: 22,
    1771: 23,
    2024: 24,
    2300: 25,
    2600: 26,
    2925: 27,
    3276: 28,
    3654: 29,
    4060: 30,
    4495: 31,
    4960: 32,
    5456: 33,
    5984: 34,
    6545: 35,
    7140: 36,
    7770: 37,
    8436: 38,
    9139: 39,
    9880: 40,
    10660: 41,
    11480: 42,
    12341: 43,
    13244: 44,
    14190: 45,
    15180: 46,
    16215: 47,
    17296: 48,
    18424: 49,
    19600: 50,
}


class RougeRewardBuilder:
    def __init__(self, base_path):
        self.base_path = os.path.join(base_path, "rouge_npy")

        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
            if self.base_path[-1] == "/":
                reading_path = self.base_path[:-1] + ".tar"
            else:
                reading_path = self.base_path + ".tar"
            with tarfile.open(reading_path) as tar:
                logging.info(
                    f"Rewards not yet extracted to {self.base_path} folder. Doing it now."
                )
                tar.extractall(base_path)

    def init_scorer(self, article_id, subset):
        assert subset in ["train", "test", "val"]

        read_path = os.path.join(self.base_path, subset, f"{article_id}.npy")

        return RougeRewardScorer(read_path)

    @staticmethod
    def from_config(config):
        return RougeRewardBuilder(config.data_path)


class RougeRewardScorer:
    def __init__(self, read_path):
        self.scores = np.load(read_path)
        self.n_sents = LEN_TO_N[self.scores.shape[0]]

    def __call__(self, summary_idxs):
        if len(summary_idxs) == 3 and all(
            [s_idx < self.n_sents for s_idx in summary_idxs]
        ):
            summary_idxs = tuple(sorted(summary_idxs))
            idx = index_getter(*summary_idxs)
            return self.scores[idx]
        else:
            return np.zeros((1,), dtype=np.float32)

    def summary_from_idx(self, index):
        for k in range(2, self.n_sents + 1):
            if comb(k + 1, 3, exact=True) > index:
                k_val = comb(k, 3, exact=True)
                for j in range(1, k):
                    if comb(j + 1, 2, exact=True) + k_val > index:
                        j_val = comb(j, 2, exact=True)
                        i = index - j_val - k_val
                        return np.array([i, j, k])


def index_getter(i, j, k):
    return comb(k, 3, exact=True) + comb(j, 2, exact=True) + comb(i, 1, exact=True)


def inverse_index_getter(idx):
    pass
