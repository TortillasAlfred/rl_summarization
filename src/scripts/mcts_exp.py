from src.domain.utils import configure_logging, set_random_seed
from src.factories.dataset import DatasetFactory
from src.factories.model import ModelFactory
from src.factories.reward import RewardFactory
from src.factories.trainer import TrainerFactory

# HACK
from src.domain.models.rlsum_oful_exp import RLSumOFULEXP

import yaml
import logging
import argparse
from test_tube import HyperOptArgumentParser
import torch


def main(_config, cluster=None):
    configure_logging()
    set_random_seed(_config.seed)

    logging.info("Beginning mcts experiment script with following config :")
    logging.info(_config)

    dataset = DatasetFactory.get_dataset(_config)
    trainer = TrainerFactory.get_trainer(_config)
    reward = RewardFactory.get_reward(_config)
    model = ModelFactory.get_model(dataset, reward, _config)

    # HACK
    # trainer.test(model)
    # trainer.fit(model)
    if _config.raw_run == 0:
        model.raw_run_done = False
        trainer.test(model)
    else:
        ckpt_paths_per_alpha = {
            1.0: "/project/def-lulam50/magod/rl_summ/exp_logging/weight_saving/rlsum_oful_248/epoch=3-val_greedy_rouge_mean=0.25333.ckpt",
            0.1: "/project/def-lulam50/magod/rl_summ/exp_logging/weight_saving/rlsum_oful_246/epoch=3-val_greedy_rouge_mean=0.30796.ckpt",
            10.0: "/project/def-lulam50/magod/rl_summ/exp_logging/weight_saving/rlsum_oful_249/epoch=3-val_greedy_rouge_mean=0.26634.ckpt",
        }
        checkpoint = torch.load(ckpt_paths_per_alpha[_config.alpha_oful])
        model.load_state_dict(checkpoint["state_dict"])
        model.raw_run_done = True
        trainer.test(model)

    logging.info("Done")


if __name__ == "__main__":
    base_configs = yaml.load(open("./configs/base.yaml"), Loader=yaml.FullLoader)
    argument_parser = HyperOptArgumentParser()
    for config, value in base_configs.items():
        if type(value) is bool:
            # Hack as per https://stackoverflow.com/a/46951029
            argument_parser.add_argument(
                "--{}".format(config),
                type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
                default=value,
            )
        else:
            argument_parser.add_argument(
                "--{}".format(config), type=type(value), default=value
            )
    options = argument_parser.parse_args()
    main(options)
