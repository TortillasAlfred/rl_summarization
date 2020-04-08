from src.domain.state import BanditExtractiveSummarizationState
from src.domain.metrics_logging import DefaultLoggedMetrics

import numpy as np


class BanditSummarizationEnvironment:
    def __init__(self, reward_builder, episode_length):
        self.reward_builder = reward_builder
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
        self.reward_scorers = [
            self.reward_builder.init_scorer(articles.id[i], subset)
            for i in range(articles.batch_size)
        ]
        self.logged_metrics = DefaultLoggedMetrics(articles.batch_size * self.n_repeats)

        return self.states

    def update(self, actions, actions_probs):
        for state, action in zip(self.states, actions):
            state.update(action)

        rewards = np.asarray(
            [
                self.reward_scorers[scorer_idx].get_score(
                    self.states[scorer_idx * self.n_repeats + state_idx].summary_idxs
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
