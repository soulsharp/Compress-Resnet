import os
import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from model.resnet_pl import Resnet50Module
from utils.utils import load_yaml


def train(args):
    seed_everything(0)
    logger = None
    if args.logger == "wandb":
        logger = WandbLogger(
            name="Resnet50Retraining",
            save_dir=args.log_path,
            project="Resnet50-Compress",
        )
    elif args.logger == "tb":
        logger = TensorBoardLogger(save_dir=args.log_path, name="Resnet50Retraining")

    assert logger is not None

    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename="best",
        monitor="acc/val",
        mode="max",
        save_last=False,
        enable_version_counter=True,
    )

    config = load_yaml(args.cfg_path)
    assert config is not None, "Config cannot be None"
    train_config = config["train"]

    model = Resnet50Module(args.num_classes, train_config, args.pretrained_path)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        accelerator=accelerator,
        logger=logger,
        deterministic=True,
        log_every_n_steps=25,
        max_epochs=args.max_epochs,
        callbacks=checkpoint,
        precision=args.precision,
    )

    trainer.fit(
        model,
    )
    trainer.test()
