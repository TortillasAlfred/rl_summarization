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
        "seeds": list(range(5)),
        "classifier_type": [
            "MLPClassifier-small",
            "MLPClassifier-medium",
            "MLPClassifier-large",
            "Transformer-small",
            "Transformer-medium",
            "Transformer-large",
        ],
    }

    configs = list(product(*grid.values()))
    config = configs[args.job_index]  # We take the one that matches our index

    # Adjusting the configuration
    args.seed = config[0]
    classifier, size = config[1]

    if classifier == "MLPClassifier":
        args.encoder = classifier
        args.MLPClassifier_size = size
    elif classifier == "Transformer":
        args.encoder = classifier
        if size == "small":
            args.num_inter_sentence_transformers, args.num_head = (1, 3)
        elif size == "medium":
            args.num_inter_sentence_transformers, args.num_head = (2, 4)
        elif size == "large":
            args.num_inter_sentence_transformers, args.num_head = (3, 5)
        else:
            raise Exception("This size doesn't exist!")
    else:
        raise NotImplementedError(f"The classifier {classifier} is not implemented!")

    return args


if __name__ == "__main__":
    base_configs = yaml.load(open("./configs/base.yaml"), Loader=yaml.FullLoader)

    if base_configs["job_index"] != -1:
        base_configs = set_config_from_index(base_configs)

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
    main(options)
