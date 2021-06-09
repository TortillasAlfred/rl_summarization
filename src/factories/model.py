from src.domain.models.bertcombisum import BertCombiSum
from src.domain.models.banditsum import *
from src.domain.models.banditsum_exp import *
from src.domain.models.rlsum_mcts_exp import *
from src.domain.models.rlsum_mcts_exp_priors import *
from src.domain.models.linsit_pretraining import LinSIT as LinSITPretraining
from src.domain.models.linsit_exp import *
from src.domain.models.linsit_exp_priors import *
from src.domain.models.linear_hypothesis_tests import *
from src.domain.models.ngrams_calc import *
from src.domain.models.sit import *
from src.domain.models.sit_priors import *
from src.domain.models.linsit import *
from src.domain.models.binary import *


class ModelFactory:
    BERTCOMBISUM = "bertcombisum"
    BANDITSUM = "banditsum"
    RLSUM_MCTS = "rlsum_mcts"
    RLSUM_MCTS_PURE = "rlsum_mcts_pure"
    BANDITUM_MCS_EXP = "banditsum_mcs_exp"
    RLSUM_MCTS_EXP = "rlsum_mcts_exp"
    RLSUM_MCTS_EXP_PRIORS = "rlsum_mcts_exp_priors"
    LINSIT_PRETRAINING = "linsit_pretraining"
    LINSIT_EXP = "linsit_exp"
    LINSIT_EXP_PRIORS = "linsit_exp_priors"
    LINEAR_HYPOTHESIS = "lin_hyp"
    NGRAMS_CALC = "ngrams_calc"
    SIT = "sit"
    SIT_PRIORS = "sit_priors"
    LINSIT = "linsit"
    BINARY = "binary"

    @classmethod
    def get_model(cls, dataset, reward, config):
        model = config.model

        if model == cls.BERTCOMBISUM:
            return BertCombiSum.from_config(dataset, reward, config)
        elif model == cls.BANDITSUM:
            return BanditSum.from_config(dataset, reward, config)
        elif model == cls.BANDITUM_MCS_EXP:
            return BanditSumMCSExperiment.from_config(dataset, reward, config)
        elif model == cls.RLSUM_MCTS_EXP:
            return RLSumMCTSEXP.from_config(dataset, reward, config)
        elif model == cls.RLSUM_MCTS_EXP_PRIORS:
            return RLSumMCTSEXPPriors.from_config(dataset, reward, config)
        elif model == cls.LINSIT_PRETRAINING:
            return LinSITPretraining.from_config(dataset, reward, config)
        elif model == cls.LINSIT_EXP:
            return LinSITExp.from_config(dataset, reward, config)
        elif model == cls.LINSIT_EXP_PRIORS:
            return LinSITExpPriors.from_config(dataset, reward, config)
        elif model == cls.LINEAR_HYPOTHESIS:
            return LinearHypothesisTests.from_config(dataset, reward, config)
        elif model == cls.NGRAMS_CALC:
            return NGramsPCA.from_config(dataset, reward, config)
        elif model == cls.SIT:
            return SITModel.from_config(dataset, reward, config)
        elif model == cls.SIT_PRIORS:
            return SITPriorsModel.from_config(dataset, reward, config)
        elif model == cls.LINSIT:
            return LinSITModel.from_config(dataset, reward, config)
        elif model == cls.BINARY:
            return BinaryModel.from_config(dataset, reward, config)
        else:
            raise ValueError(f"Model {model} not implemented.")
