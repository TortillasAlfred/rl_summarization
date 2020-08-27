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
        overfit_pct,
        val_check_interval,
        default_save_path,
        weights_save_path,
        model,
        max_epochs,
        config,
        cluster=None,
    ):
        if hasattr(config, "hpc_exp_number"):
            name = f"{model}_{config.hpc_exp_number}"
        else:
            name = model

        weights_save_path = "/".join([weights_save_path, name])
        checkpoint_callback = ModelCheckpoint(
            filepath="/".join(
                [weights_save_path, "{epoch}-{val_greedy_rouge_mean:.5f}"]
            ),
            save_top_k=3,
            verbose=True,
            monitor="val_greedy_rouge_mean",
            mode="max",
            prefix="",
        )
        logger = TestTubeLogger(
            save_dir=default_save_path, name=name, create_git_tag=True
        )

        if hasattr(config, "test_tube_slurm_cmd_path"):
            self.launch_script_path = config.test_tube_slurm_cmd_path

        super(PytorchLightningTrainer, self).__init__(
            gradient_clip_val=gradient_clip_val,
            gpus=gpus,
            fast_dev_run=fast_dev_run,
            distributed_backend=distributed_backend,
            overfit_pct=overfit_pct,
            val_check_interval=val_check_interval,
            default_root_dir=default_save_path,
            weights_save_path=weights_save_path,
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            max_epochs=max_epochs,
            max_steps=config.warmup_batches,  # HACK FOR MCTS EXP
        )

    @staticmethod
    def from_config(config):
        return PytorchLightningTrainer(
            config.gradient_clip_val,
            config.gpus,
            config.fast_dev_run,
            config.distributed_backend,
            config.overfit_pct,
            config.val_check_interval,
            config.default_save_path,
            config.weights_save_path,
            config.model,
            config.max_epochs,
            config,
        )
