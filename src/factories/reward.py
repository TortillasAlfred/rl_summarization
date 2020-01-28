from src.domain.rewards.rouge import RougeReward


class RewardFactory:
    ROUGE = 'rouge'

    @classmethod
    def get_reward(cls, dataset, config):
        reward_type = config['reward']

        if reward_type == cls.ROUGE:
            return RougeReward.from_config(dataset, config)
        else:
            raise ValueError(f"Reward type {reward_type} not implemented.")
