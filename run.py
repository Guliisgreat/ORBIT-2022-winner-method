import os
from pathlib import Path
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.utilities.cloud_io import load as pl_load

from pytorchlightning_trainer.utils.common import log_hyperparameters, PROJECT_ROOT
from pytorchlightning_trainer.callbacks.tensorboard_weight_distribution import TensorboardModelDistribution
from pytorchlightning_trainer.callbacks.unfreeze_backbone import FeatureExtractorFreezeUnfreeze


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info(f"Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info(f"Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info(f"Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    if "TensorboardModelDistribution" in cfg.train:
        hydra.utils.log.info(f"Adding callback <ModelCheckpoint>")
        callbacks.append(
            TensorboardModelDistribution()
        )
    if "unfreeze_backbone_epoch" in cfg.model:
        hydra.utils.log.info(f"Adding callback < FeatureExtractorFreezeUnfreeze>")
        callbacks.append(FeatureExtractorFreezeUnfreeze(cfg.model.unfreeze_backbone_epoch))

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """

    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing

        # cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers = 0
        cfg.data.datamodule.num_workers = 0
        cfg.data.datamodule.num_workers = 0

        # # Switch wandb mode to offline to prevent online logging
        # cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
    )
    if "load_pretrained" in cfg.train:
        hydra.utils.log.info(f"Manually load pretrained weights ...(bugs in default pytorch_lightning)")
        hydra.utils.log.info(cfg.train.load_pretrained)
        checkpoint = pl_load(path_or_url=cfg.train.load_pretrained)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        # model.model.load_state_dict(checkpoint, strict=True)

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    if "tensorboard" in cfg.logging:
        exp_folder = os.path.join(PROJECT_ROOT, cfg.logging.tensorboard.logger_dir, cfg.train.exp_name)
        if not os.path.exists(
                Path(exp_folder).parent):
            os.makedirs(exp_folder)
            hydra.utils.log.info("create experiment log directory {}".format(exp_folder))

        hydra.utils.log.info(f"Instantiating <TensorboardLogger>")
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(PROJECT_ROOT, cfg.logging.tensorboard.logger_dir),
            name=cfg.train.exp_name,
            default_hp_metric=False
        )

    hydra.utils.log.info(f"Instantiating the Trainer")

    # The Lightning core, the Trainer
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=tb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        val_check_interval=cfg.logging.val_check_interval,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        # profiler="simple",
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    if cfg.train.skip_training:
        hydra.utils.log.info(f"Skip training!")
        hydra.utils.log.info(f"Starting testing!")
        trainer.test(model=model, datamodule=datamodule)
    else:
        hydra.utils.log.info(f"Starting training!")
        trainer.fit(model=model, datamodule=datamodule)
        hydra.utils.log.info(f"Starting testing!")
        trainer.test(datamodule=datamodule)


@hydra.main(config_path=str(PROJECT_ROOT / "pytorchlightning_trainer/conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
