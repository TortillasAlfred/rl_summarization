from collections import defaultdict, namedtuple
import torch


class TextDataCollator:
    def __init__(self, reward_builder, subset):
        self.reward_builder = reward_builder
        self.subset = subset

    def __call__(self, data):
        batch = defaultdict(list)

        for datum in data:
            for name, field in datum.items():
                batch[name].append(field)

        batch["scorers"] = get_reward_scorers(
            self.reward_builder, batch["id"], self.subset
        )

        tensor_keys = [k for k, d in batch.items() if isinstance(d[0], torch.Tensor)]

        for key in tensor_keys:
            batch[key] = torch.stack(batch[key])

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
