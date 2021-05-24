from src.domain.dataset import CnnDailyMailDataset
from src.domain.dataset_bert import CnnDailyMailDatasetBert


class DatasetFactory:
    CNN_DAILYMAIL = "cnn_dailymail"
    CNN_DAILYMAIL_BERT = "cnn_dailymail_bert"

    @classmethod
    def get_dataset(cls, config):
        dataset_type = config.dataset

        if dataset_type == cls.CNN_DAILYMAIL_BERT:
            return CnnDailyMailDatasetBert.from_config(config)
        elif dataset_type == cls.CNN_DAILYMAIL:
            return CnnDailyMailDataset.from_config(config)
        else:
            raise ValueError(f"Dataset type {dataset_type} not implemented.")
