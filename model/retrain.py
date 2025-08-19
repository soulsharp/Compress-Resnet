import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from model.resnet_pl import Resnet50Module
from utils.utils import load_yaml

def main(args):
    seed_everything(0)
    if args.logger == "wandb":
        logger = WandbLogger(name="Resnet50Retraining", save_dir=args.log_path, project="Resnet50-Compress")
    else:
        logger = TensorBoardLogger(save_dir=args.log_path, name="Resnet50Retraining")

    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename="best",
        monitor="acc/val", 
        mode="max", 
        save_last=False,
        enable_version_counter=True
        )

    config = load_yaml(args.cfg_path)
    assert config is not None, "Config cannot be None"
    train_config = config["train"]

    model = Resnet50Module(args.num_classes, train_config, args.pretrained_path)
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        accelerator=accelerator,
        fast_dev_run=bool(args.dev),
        logger=logger,
        deterministic=True,
        log_every_n_steps=25,
        max_epochs=args.max_epochs,
        callbacks=checkpoint,
        precision=args.precision,
    )

    if bool(args.pretrained):
        state_dict = os.path.join(
            "cifar10_models", "state_dicts", args.classifier + ".pt"
        )
        model.model.load_state_dict(torch.load(state_dict))

    if bool(args.test_phase):
        trainer.test(model, data.test_dataloader())
    else:
        trainer.fit(model, data)
        trainer.test()