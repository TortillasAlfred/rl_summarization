from collections import defaultdict, namedtuple


def text_data_collator(fields, reward_builder, subset):
    def collate(fields, reward_builder, subset, data):
        batch = defaultdict(list)

        for datum in data:
            for name, field in fields.items():
                batch[name].append(field.preprocess(getattr(datum, name)))

        batch = {name: field.process(batch[name]) for name, field in fields.items()}
        batch["scorers"] = get_reward_scorers(reward_builder, batch["id"], subset)

        # batch = namedtuple("batch", batch.keys())(**batch)

        return batch

    return lambda d: collate(fields, reward_builder, subset, d)


def get_reward_scorers(reward_builder, ids, subset):
    if subset in ["train", "val", "test"]:
        return [reward_builder.init_scorer(id, subset) for id in ids]
    # elif subset in ["val", "test"]:
    #     return [RougePythonReward() for _ in range(batch_size)]
    else:
        raise ValueError(
            f'Bad subset : {subset}. Should be one of ["train", "val", "test].'
        )
