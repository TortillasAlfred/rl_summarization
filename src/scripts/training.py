from src.domain.utils import configure_logging, set_random_seed
from src.factories.dataset import DatasetFactory
from src.factories.model import ModelFactory
from src.factories.reward import RewardFactory
from src.factories.trainer import TrainerFactory

from itertools import product

import yaml
import logging
import argparse
from test_tube import HyperOptArgumentParser
import os


def main(_config, cluster=None):
    configure_logging()
    set_random_seed(_config.seed)

    if "$SLURM_TMPDIR" in _config.data_path:
        _config.data_path = _config.data_path.replace("$SLURM_TMPDIR", os.environ["SLURM_TMPDIR"])

    logging.info("Beginning Training script with following config :")
    logging.info(_config)

    dataset = DatasetFactory.get_dataset(_config)
    trainer = TrainerFactory.get_trainer(_config)
    reward = RewardFactory.get_reward(_config)
    model = ModelFactory.get_model(dataset, reward, _config)

    trainer.fit(model)
    trainer.test()

    logging.info("Done")


def set_config_from_index(args):
    grid = {
        "seed": list(range(5)),
        "encoder": [
            "MLPClassifier",
            "Transformer",
        ],
        "encoder_size": ["small", "med", "large"],
        "ucb_sampling": ["fix", "linear"],
        "rescale_targets": [True, False],
    }

    configs = list(product(*grid.values()))
    config = configs[args.job_index]  # We take the one that matches our index

    # Adjusting the configuration
    args.seed, args.encoder, args.encoder_size, args.ucb_sampling, args.rescale_targets = config

    return args


if __name__ == "__main__":
    base_configs = yaml.load(open("./configs/base.yaml"), Loader=yaml.FullLoader)

    argument_parser = HyperOptArgumentParser()
    for config, value in base_configs.items():
        if type(value) is bool:
            # Hack as per https://stackoverflow.com/a/46951029
            argument_parser.add_argument(
                "--{}".format(config), type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=value
            )
        else:
            argument_parser.add_argument("--{}".format(config), type=type(value), default=value)

    options = argument_parser.parse_args()

    if options.job_index != -1:
        options = set_config_from_index(options)

    main(options)
