import torch


class TextDataCollator:
    def __init__(self, fields, reward_builder, subset):
        self.fields = fields
        self.reward_builder = reward_builder
        self.subset = subset

    def __call__(self, data):
        batch = {name: f.process([d[name] for d in data]) for name, f in self.fields}

        batch["scorers"] = get_reward_scorers(
            self.reward_builder, batch["id"], self.subset
        )

        return list(batch.values())


def get_reward_scorers(reward_builder, ids, subset):
    if subset in ["train", "val", "test"]:
        return [reward_builder.init_scorer(id, subset) for id in ids]
    # elif subset in ["val", "test"]:
    #     return [RougePythonReward() for _ in range(batch_size)]
    else:
        raise ValueError(
            f'Bad subset : {subset}. Should be one of ["train", "val", "test].'
        )


class NGRAMS:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, doc_contents):
        indices = torch.LongTensor(
            [
                [i, w]
                for i, sent in enumerate(doc_contents)
                for w in sent
                if w != self.pad_idx
            ]
        )
        values = torch.ones(indices.shape[0])
        ngrams = torch.sparse.FloatTensor(
            indices.t(), values, (indices[-1][0] + 1, doc_contents.max().cpu() + 1)
        ).to(doc_contents.device)
        u, v, _ = torch.svd_lowrank(ngrams, q=len(doc_contents))

        return u * v
