from src.domain.rewards.rouge import RougeRewardBuilder


class RewardFactory:
    ROUGE = "rouge"

    @classmethod
    def get_reward(cls, config):
        reward_type = config["reward"]

        if reward_type == cls.ROUGE:
            return RougeRewardBuilder.from_config(config)
        else:
            raise ValueError(f"Reward type {reward_type} not implemented.")
