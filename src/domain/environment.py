from src.domain.state import BanditExtractiveSummarizationState
from src.domain.metrics_logging import DefaultLoggedMetrics
from src.domain.rewards.rouge_python import RougePythonReward
from src.domain.rewards.rouge_pearl import RougePearlReward

import numpy as np
import torch


class BanditSummarizationEnvironment:
    def __init__(self, reward_builder, episode_length):
        self.train_reward_builder = reward_builder
        self.episode_length = episode_length

        self.n_repeats = 1

    def init(self, raw_abstracts, raw_contents, ids, subset, n_repeats=1):
        batch_size = len(raw_abstracts)
        self.n_repeats = n_repeats
        self.states = [
            BanditExtractiveSummarizationState(
                raw_contents[i], raw_abstracts[i], self.episode_length
            )
            for i in range(batch_size)
            for _ in range(self.n_repeats)
        ]
        self.reward_scorers = self.__get_reward_scorers(ids, batch_size, subset)
        self.logged_metrics = DefaultLoggedMetrics(batch_size * self.n_repeats)

        return self.states

    def soft_init(self, raw_contents, raw_abstracts, subset, n_repeats=1):
        batch_size = len(raw_abstracts)
        self.states = [
            BanditExtractiveSummarizationState(
                raw_contents[i], raw_abstracts[i], self.episode_length,
            )
            for i in range(batch_size)
            for _ in range(self.n_repeats)
        ]

        return self.states

    def __get_reward_scorers(self, ids, batch_size, subset):
        if subset == "train":
            return [
                self.train_reward_builder.init_scorer(ids[i], subset)
                for i in range(batch_size)
            ]
        elif subset in ["val", "test"]:
            return [RougePythonReward() for _ in range(batch_size)]
        else:
            raise ValueError(
                f'Bad subset : {subset}. Should be one of ["train", "val", "test].'
            )

    def update(
        self, actions, actions_probs=None, is_mcts=False, policy=None, q_vals=None,
    ):
        batch_size = len(actions)
        gpu_device_idx = torch.cuda.current_device()

        print(gpu_device_idx)

        for i in range(batch_size):
            state = self.states[gpu_device_idx * batch_size + i]
            state.update(actions[i])

        rewards = self.get_scores(f"cuda:{gpu_device_idx}", batch_size, gpu_device_idx)

        self.logged_metrics.log(
            actions, actions_probs, rewards, is_mcts, self.done(), policy, q_vals,
        )

        return self.states, rewards

    def get_scores(self, device, batch_size, gpu_device_idx):
        return torch.tensor(
            [
                self.reward_scorers[gpu_device_idx * batch_size + idx].get_score(
                    self.states[gpu_device_idx * batch_size + idx]
                )
                for idx in range(batch_size)
            ],
            device=device,
        )

    def done(self):
        return all([state.done for state in self.states])

    def get_logged_metrics(self):
        return self.logged_metrics.to_dict()
