from src.domain.models.baselines import *
from src.domain.models.banditsum import *
from src.domain.models.banditsum_mcts import *
from src.domain.models.rlsum_mcts import *
from src.domain.models.rlsum_oh import *
from src.domain.models.azsum_mcts import *
from src.domain.models.a2c import *


class ModelFactory:
    LEAD3 = "lead3"
    BANDITSUM = "banditsum"
    A2C = "a2c"
    BANDITSUM_MCTS = "banditsum_mcts"
    RLSUM_MCTS = "rlsum_mcts"
    AZSUM_MCTS = "azsum_mcts"
    RLSUM_OH = "rlsum_oh"

    @classmethod
    def get_model(cls, dataset, reward, config):
        model = config.model

        if model == cls.LEAD3:
            return Lead3.from_config(dataset, reward, config)
        elif model == cls.BANDITSUM:
            return BanditSum.from_config(dataset, reward, config)
        elif model == cls.A2C:
            return A2C.from_config(dataset, reward, config)
        elif model == cls.BANDITSUM_MCTS:
            return BanditSumMCTS.from_config(dataset, reward, config)
        elif model == cls.RLSUM_MCTS:
            return RLSumMCTS.from_config(dataset, reward, config)
        elif model == cls.AZSUM_MCTS:
            return AZSumMCTS.from_config(dataset, reward, config)
        elif model == cls.RLSUM_OH:
            return RLSumOH.from_config(dataset, reward, config)
        else:
            raise ValueError(f"Model {model} not implemented.")
