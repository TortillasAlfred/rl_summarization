# from src.domain.models.rlsum_mcts import *
# from src.domain.models.rlsum_mcts_pure import *
from src.domain.models.rlsum_oful import *
from src.domain.models.rlsum_oful_exp import *
from src.domain.models.banditsum import *
from src.domain.models.banditsum_exp import *
from src.domain.models.rlsum_mcts_exp import *
from src.domain.models.rlsum_mcts_exp_priors import *
from src.domain.models.linsit_pretraining import LinSIT as LinSITPretraining


class ModelFactory:
    BANDITSUM = "banditsum"
    RLSUM_MCTS = "rlsum_mcts"
    RLSUM_MCTS_PURE = "rlsum_mcts_pure"
    RLSUM_OFUL = "rlsum_oful"
    RLSUM_OFUL_EXP = "rlsum_oful_exp"
    BANDITUM_MCS_EXP = "banditsum_mcs_exp"
    RLSUM_MCTS_EXP = "rlsum_mcts_exp"
    RLSUM_MCTS_EXP_PRIORS = "rlsum_mcts_exp_priors"
    LINSIT_PRETRAINING = "linsit_pretraining"

    @classmethod
    def get_model(cls, dataset, reward, config):
        model = config.model

        if model == cls.BANDITSUM:
            return BanditSum.from_config(dataset, reward, config)
        # elif model == cls.RLSUM_MCTS:
        #     return RLSumMCTS.from_config(dataset, reward, config)
        # elif model == cls.RLSUM_MCTS_PURE:
        #     return RLSumMCTSPure.from_config(dataset, reward, config)
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
        elif model == cls.LINSIT_PRETRAINING:
            return LinSITPretraining.from_config(dataset, reward, config)
        else:
            raise ValueError(f"Model {model} not implemented.")
