import yaml
from torch.utils.data import DataLoader
from test_tube import HyperOptArgumentParser
from tqdm import tqdm
import numpy as np
import torch

from src.domain.dataset_bert import DatasetBertWrapper
from src.domain.loader_utils import TextDataCollator
from src.domain.rewards.rouge import RougeRewardBuilder


def main(options):
    dataset = DatasetBertWrapper.from_config(options)
    reward = RougeRewardBuilder.from_config(options)

    splits = dataset.get_splits()

    for split_name, subset in splits.items():
        word_counts = {"the": [], "animal": [], "house": [], "criminal": [], "have": [], "very": []}

        data_loader = DataLoader(
            subset,
            collate_fn=TextDataCollator(dataset.fields, reward, subset=split_name),
            batch_size=1,  # Hardcoded to 1 because more convenient by default
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        for batch in tqdm(data_loader, f"Iterating over the {split_name} dataset"):
            ids, contents, abstracts, raw_contents, raw_abstracts, scorers = batch

            valid_sentences = contents["mark_clss"]
            sentence_gap = torch.tensor(contents["sentence_gap"], dtype=torch.long, device=valid_sentences.device)
            available_sent_idxs = (torch.arange(valid_sentences.float().sum(-1).item()) + sentence_gap)[valid_sentences]
            available_sent_idxs = available_sent_idxs.long().tolist()

            available_sents = [raw_contents[0][idx] for idx in available_sent_idxs]

            for word in word_counts:
                avg_occurences = sum([word in sent for sent in available_sents]) / len(available_sents)
                word_counts[word].append(avg_occurences)

        for word, word_avgs in word_counts.items():
            print(
                f"For the {split_name} dataset, the word '{word}' is seen on average in {np.mean(word_avgs) * 100:.2f} % of the available sentences"
            )

    print("Done !")


if __name__ == "__main__":
    base_configs = yaml.load(open("./configs/base.yaml"), Loader=yaml.FullLoader)
    argument_parser = HyperOptArgumentParser()
    for config, value in base_configs.items():
        if type(value) is bool:
            # Hack as per https://stackoverflow.com/a/46951029
            argument_parser.add_argument(
                "--{}".format(config),
                type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
                default=value,
            )
        else:
            argument_parser.add_argument("--{}".format(config), type=type(value), default=value)
    options = argument_parser.parse_args()

    main(options)
