from collections import defaultdict, namedtuple
import torch


class TextDataCollator:
    def __init__(self, fields, reward_builder, subset):
        self.fields = fields
        self.reward_builder = reward_builder
        self.subset = subset

    def __call__(self, data):
        batch = {
            name: f.process([d[name] for d in data]) for name, f in self.fields
        }

        batch["scorers"] = get_reward_scorers(
            self.reward_builder, batch["id"], self.subset
        )

        return batch


def get_reward_scorers(reward_builder, ids, subset):
    if subset in ["train", "val", "test"]:
        return [reward_builder.init_scorer(id, subset) for id in ids]
    # elif subset in ["val", "test"]:
    #     return [RougePythonReward() for _ in range(batch_size)]
    else:
        raise ValueError(
            f'Bad subset : {subset}. Should be one of ["train", "val", "test].'
        )
