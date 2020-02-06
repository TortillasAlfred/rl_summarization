from src.domain.dataset import *


class DatasetFactory:
    CNN_DAILYMAIL = "cnn_dailymail"

    @classmethod
    def get_dataset(cls, config):
        dataset_type = config["dataset"]

        if dataset_type == cls.CNN_DAILYMAIL:
            return CnnDailyMailDataset.from_config(config)
        else:
            raise ValueError(f"Dataset type {dataset_type} not implemented.")
