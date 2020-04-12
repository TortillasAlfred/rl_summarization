import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger


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
        gradient_clip_val,
        gpus,
        fast_dev_run,
        distributed_backend,
        use_amp,
        overfit_pct,
        print_nan_grads,
        val_check_interval,
        default_save_path,
        weights_save_path,
        model,
        max_epochs,
        cluster=None,
    ):
        checkpoint_callback = ModelCheckpoint(
            filepath="/".join(
                [weights_save_path, model, "{epoch}-{val_greedy_rouge_mean:.5f}"]
            ),
            save_top_k=10,
            verbose=True,
            monitor="val_greedy_rouge_mean",
            mode="max",
            prefix="",
        )
        logger = TestTubeLogger(
            save_dir=default_save_path, name=model, create_git_tag=True
        )
        super(PytorchLightningTrainer, self).__init__(
            gradient_clip_val=gradient_clip_val,
            gpus=gpus,
            fast_dev_run=fast_dev_run,
            distributed_backend=distributed_backend,
            use_amp=use_amp,
            overfit_pct=overfit_pct,
            print_nan_grads=print_nan_grads,
            val_check_interval=val_check_interval,
            default_save_path=default_save_path,
            weights_save_path=weights_save_path,
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            max_epochs=max_epochs,
            cluster=cluster,
        )

    @staticmethod
    def from_config(config, cluster):
        return PytorchLightningTrainer(
            config.gradient_clip_val,
            config.gpus,
            config.fast_dev_run,
            config.distributed_backend,
            config.use_amp,
            config.overfit_pct,
            config.print_nan_grads,
            config.val_check_interval,
            config.default_save_path,
            config.weights_save_path,
            config.model,
            config.max_epochs,
            cluster,
        )
