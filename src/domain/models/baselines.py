from src.domain.utils import datetime_tqdm

import logging
import torch
from torchtext.data import BucketIterator


class Lead3:
    def __init__(self, dataset, reward):
        self.dataset = dataset
        self.reward = reward

    def train(self):
        logging.info("No training needed for LEAD-3 baseline.")

    def test(self):
        test_split = self.dataset.get_splits()['test']
        test_loader = BucketIterator(
            test_split,
            train=False,
            batch_size=64,
            shuffle=False,
            sort=False,
            device='cpu',
        )

        all_rewards = []

        for inputs, targets in datetime_tqdm(
            test_loader, desc="Processing test samples"
        ):
            raw_contents, contents = inputs
            raw_abstracts, abstracts = targets

            selected_idxs = self(contents, raw_contents)
            summaries = [
                [content[i] for i in idxs]
                for content, idxs in zip(raw_contents, selected_idxs)
            ]

            all_rewards.append(self.reward(summaries, raw_abstracts, 'cpu'))

        all_rewards = torch.cat(all_rewards)
        logging.info(f"Max reward : {all_rewards.max(dim=0)}")
        logging.info(f"Min reward : {all_rewards.min(dim=0)}")
        logging.info(f"Mean reward : {all_rewards.mean(dim=0)}")

    def __call__(self, contents, raw_contents):
        return [range(min(3, len(c))) for c in raw_contents]

    @staticmethod
    def from_config(dataset, reward, config):
        return Lead3(dataset, reward)
