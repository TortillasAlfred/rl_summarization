from src.domain.models.baselines import *
from src.domain.models.banditsum import *
from src.domain.models.banditsum_mcts import *
from src.domain.models.rlsum_mcts import *
from src.domain.models.rlsum_mcts_pure import *
from src.domain.models.rlsum_value_pure import *
from src.domain.models.rlsum_value import *
from src.domain.models.rlsum_value_inference import *
from src.domain.models.rlsum_value_pure_inference import *
from src.domain.models.rlsum_oh import *
from src.domain.models.rlsum_oful import *
from src.domain.models.rlsum_oful_exp import *
from src.domain.models.azsum_mcts import *
from src.domain.models.a2c import *
from src.domain.models.banditsum_exp import *
from src.domain.models.rlsum_mcts_exp import *
from src.domain.models.rlsum_mcts_exp_priors import *


class ModelFactory:
    LEAD3 = "lead3"
    BANDITSUM = "banditsum"
    A2C = "a2c"
    BANDITSUM_MCTS = "banditsum_mcts"
    RLSUM_MCTS = "rlsum_mcts"
    AZSUM_MCTS = "azsum_mcts"
    RLSUM_OH = "rlsum_oh"
    RLSUM_MCTS_PURE = "rlsum_mcts_pure"
    RLSUM_VALUE_PURE = "rlsum_value_pure"
    RLSUM_VALUE = "rlsum_value"
    RLSUM_VALUE_INFERENCE = "rlsum_value_inference"
    RLSUM_VALUE_PURE_INFERENCE = "rlsum_value_pure_inference"
    RLSUM_OFUL = "rlsum_oful"
    RLSUM_OFUL_EXP = "rlsum_oful_exp"
    BANDITUM_MCS_EXP = "banditsum_mcs_exp"
    RLSUM_MCTS_EXP = "rlsum_mcts_exp"
    RLSUM_MCTS_EXP_PRIORS = "rlsum_mcts_exp_priors"

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
        elif model == cls.RLSUM_MCTS_PURE:
            return RLSumMCTSPure.from_config(dataset, reward, config)
        elif model == cls.RLSUM_VALUE_PURE:
            return RLSumValuePure.from_config(dataset, reward, config)
        elif model == cls.RLSUM_VALUE:
            return RLSumValue.from_config(dataset, reward, config)
        elif model == cls.RLSUM_VALUE_INFERENCE:
            return RLSumValueInference.from_config(dataset, reward, config)
        elif model == cls.RLSUM_VALUE_PURE_INFERENCE:
            return RLSumValuePureInference.from_config(dataset, reward, config)
        elif model == cls.RLSUM_OFUL:
            return RLSumOFUL.from_config(dataset, reward, config)
        elif model == cls.RLSUM_OFUL_EXP:
            return RLSumOFULEXP.from_config(dataset, reward, config)
        elif model == cls.BANDITUM_MCS_EXP:
            return BanditSumMCSExperiment.from_config(dataset, reward, config)
        elif model == cls.RLSUM_MCTS_EXP:
            return RLSumMCTSEXP.from_config(dataset, reward, config)
        elif model == cls.RLSUM_MCTS_EXP_PRIORS:
            return RLSumMCTSEXPPriors.from_config(dataset, reward, config)
        else:
            raise ValueError(f"Model {model} not implemented.")
