from src.domain.models.baselines import *


class ModelFactory:
    LEAD3 = 'lead3'

    @classmethod
    def get_model(cls, dataset, reward, config):
        model = config['model']

        if model == cls.LEAD3:
            return Lead3.from_config(dataset, reward, config)
        else:
            raise ValueError(f"Model {model} not implemented.")
