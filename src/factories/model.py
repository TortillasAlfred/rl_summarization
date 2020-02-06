from src.domain.models.baselines import *
from src.domain.models.bandit import *


class ModelFactory:
    LEAD3 = "lead3"
    BANDITSUM = "banditsum"

    @classmethod
    def get_model(cls, dataset, reward, config):
        model = config["model"]

        if model == cls.LEAD3:
            return Lead3.from_config(dataset, reward, config)
        elif model == cls.BANDITSUM:
            return BanditSum.from_config(dataset, reward, config)
        else:
            raise ValueError(f"Model {model} not implemented.")
