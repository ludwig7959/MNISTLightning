
import argparse

import hydra
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from datasets import MNISTDataModule

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def train(cfg):

    if cfg.seed.enabled:
        seed_everything(cfg.seed.value)

    datamodule = MNISTDataModule(
        train_batch_size=cfg.train.batch_size,
        val_batch_size=cfg.train.validation.batch_size,
        train_workers=cfg.train.num_workers,
        val_workers=cfg.train.validation.num_workers
    )

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    tensorboard_logger = TensorBoardLogger(save_dir="tb_logs", name=model.__class__.__name__)

    callbacks = []
    for callback in cfg.train.callbacks:
        callbacks.append(hydra.utils.instantiate(callback))

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        accelerator=cfg.train.accelerator.type,
        devices=cfg.train.accelerator.devices,
        callbacks=callbacks,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=100,
        logger=tensorboard_logger
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
