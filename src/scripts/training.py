from src.domain.utils import configure_logging, set_random_seed
from src.factories.dataset import DatasetFactory
from src.factories.model import ModelFactory
from src.factories.reward import RewardFactory
from src.factories.trainer import TrainerFactory

import yaml
import logging
import argparse


def main(_config):
    configure_logging()
    set_random_seed(_config["seed"])

    logging.info("Beginning training script with following config :")
    logging.info(_config)

    dataset = DatasetFactory.get_dataset(_config)
    trainer = TrainerFactory.get_trainer(_config)
    reward = RewardFactory.get_reward(_config)
    model = ModelFactory.get_model(dataset, reward, _config)

    trainer.fit(model)
    trainer.test(model)

    logging.info("Done")


if __name__ == "__main__":
    base_configs = yaml.load(open("./configs/base.yaml"), Loader=yaml.FullLoader)
    argument_parser = argparse.ArgumentParser()
    for config, value in base_configs.items():
        if type(value) is bool:
            # Hack as per https://stackoverflow.com/a/46951029
            argument_parser.add_argument(
                "--{}".format(config),
                type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
                default=value,
            )
        elif type(value) is list:
            argument_parser.add_argument(
                "--{}".format(config), nargs="+", default=value
            )
        else:
            argument_parser.add_argument(
                "--{}".format(config), type=type(value), default=value
            )
    options = argument_parser.parse_args()
    configs = vars(options)
    main(configs)
