from src.domain.trainer import *


class TrainerFactory:
    GRADIENT_FREE = "gradient_free"
    PYTORCH_LIGHTNING = "pl"

    @classmethod
    def get_trainer(cls, config):
        trainer_type = config.trainer

        if trainer_type == cls.GRADIENT_FREE:
            return GradientFreeTrainer.from_config(config)
        elif trainer_type == cls.PYTORCH_LIGHTNING:
            return PytorchLightningTrainer.from_config(config)
        else:
            raise ValueError(f"Trainer type {trainer_type} not implemented.")
