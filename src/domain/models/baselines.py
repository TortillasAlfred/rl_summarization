from src.domain.utils import datetime_tqdm

import logging
import torch


class Lead3:
    def __init__(self, dataset, reward):
        self.dataset = dataset
        self.reward = reward

    def train(self):
        logging.info('No training needed for LEAD-3 baseline.')

    def test(self):
        test_loader = self.dataset.get_loaders(batch_size=10,
                                               device='cuda:0',
                                               shuffle=False)['test']

        all_rewards = []

        for inputs, targets in datetime_tqdm(test_loader,
                                             desc='Processing test samples'):
            raw_contents, contents = inputs
            raw_abstracts, abstracts = targets

            selected_idxs = self(contents, raw_contents)
            summaries = [[content[i] for i in idxs]
                         for content, idxs in zip(raw_contents, selected_idxs)]

            all_rewards.append(self.reward(summaries, raw_abstracts))

        all_rewards = torch.cat(all_rewards)
        logging.info(f'Max reward : {all_rewards.max(dim=0)}')
        logging.info(f'Min reward : {all_rewards.min(dim=0)}')
        logging.info(f'Mean reward : {all_rewards.mean(dim=0)}')

    def __call__(self, contents, raw_contents):
        return [range(min(3, len(c))) for c in raw_contents]

    @staticmethod
    def from_config(dataset, reward, config):
        return Lead3(dataset, reward)