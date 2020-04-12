from src.domain.state import BanditExtractiveSummarizationState
from src.domain.metrics_logging import DefaultLoggedMetrics
from src.domain.rewards.rouge_python import RougePythonReward
from src.domain.rewards.rouge_pearl import RougePearlReward

import numpy as np


class BanditSummarizationEnvironment:
    def __init__(self, reward_builder, episode_length):
        self.train_reward_builder = reward_builder
        self.episode_length = episode_length

        self.n_repeats = 1
        self.done_steps = 0

    def init(self, articles, subset, n_repeats=1):
        self.n_repeats = n_repeats
        self.states = [
            BanditExtractiveSummarizationState(
                articles.content[i], articles.raw_content[i], articles.raw_abstract[i],
            )
            for i in range(articles.batch_size)
            for _ in range(self.n_repeats)
        ]
        self.reward_scorers = self.__get_reward_scorers(articles, subset)
        self.logged_metrics = DefaultLoggedMetrics(articles.batch_size * self.n_repeats)

        return self.states

    def __get_reward_scorers(self, articles, subset):
        if subset == "train":
            return [
                self.train_reward_builder.init_scorer(articles.id[i], subset)
                for i in range(articles.batch_size)
            ]
        elif subset in ["val", "test"]:
            return [RougePythonReward() for _ in range(articles.batch_size)]
        else:
            raise ValueError(
                f'Bad subset : {subset}. Should be one of ["train", "val", "test].'
            )

    def update(self, actions, actions_probs):
        for state, action in zip(self.states, actions):
            state.update(action)

        rewards = np.asarray(
            [
                self.reward_scorers[scorer_idx].get_score(
                    self.states[scorer_idx * self.n_repeats + state_idx]
                )
                for state_idx in range(self.n_repeats)
                for scorer_idx in range(len(self.reward_scorers))
            ]
        )
        self.logged_metrics.log(actions, actions_probs, rewards)
        self.done_steps += 1

        return self.states, rewards

    def done(self):
        return self.done_steps == self.episode_length or all(
            [state.done for state in self.states]
        )

    def get_logged_metrics(self):
        return vars(self.logged_metrics)
