import logging
import os
from pytorch_lightning import Trainer


class GradientFreeTrainer:
    def fit(self, model):
        logging.info("Begin model training")
        model.train()
        logging.info("Model training done")

    def test(self, model):
        logging.info("Begin model testing")
        model.test()
        logging.info("Model testing done")

    @staticmethod
    def from_config(config):
        return GradientFreeTrainer()


class PytorchLightningTrainer(Trainer):
    def __init__(
        self,
        save_path,
        gradient_clip_val,
        gpus,
        fast_dev_run,
        max_epochs,
        distributed_backend,
        use_amp,
        overfit_pct,
    ):
        super(PytorchLightningTrainer, self).__init__(
            early_stop_callback=False,
            default_save_path=save_path,
            weights_save_path=os.path.join(save_path, "weights"),
            gradient_clip_val=gradient_clip_val,
            gpus=gpus,
            fast_dev_run=fast_dev_run,
            max_nb_epochs=max_epochs,
            distributed_backend=distributed_backend,
            use_amp=use_amp,
            overfit_pct=overfit_pct,
        )

    @staticmethod
    def from_config(config):
        return PytorchLightningTrainer(
            config["save_path"],
            config["gradient_clip_val"],
            config["gpus"],
            config["fast_dev_run"],
            config["max_epochs"],
            config["distributed_backend"],
            config["use_amp"],
            config["overfit_pct"],
        )
