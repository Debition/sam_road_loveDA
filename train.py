from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config
from loveda_dataset import create_loveda_dataloaders
from sam_loveda import SAMLoveDA

import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor


parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the default config. See /config for examples.",
)
parser.add_argument(
    "--resume", default=None, help="checkpoint of the last epoch of the model"
)
parser.add_argument(
    "--precision", default=16, help="32 or 16"
)
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--dev_run", default=False, action='store_true'
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    dev_run = args.dev_run or args.fast_dev_run

    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="sam_loveda",
        # track hyperparameters and run metadata
        config=config,
        # disable wandb if debugging
        mode='disabled' if dev_run else None
    )


    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    

    net = SAMLoveDA(config)

    # 使用新的create_loveda_dataloaders函数创建数据加载器
    train_loader, val_loader, _ = create_loveda_dataloaders(config)

    if dev_run:
        # 限制数据集大小用于快速测试
        subset_size = 10
        train_indices = list(range(min(subset_size, len(train_loader.dataset))))
        val_indices = list(range(min(subset_size, len(val_loader.dataset))))
        
        train_subset = torch.utils.data.Subset(train_loader.dataset, train_indices)
        val_subset = torch.utils.data.Subset(val_loader.dataset, val_indices)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATA_WORKER_NUM,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA_WORKER_NUM,
            pin_memory=True,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get('CHECKPOINT_DIR', 'checkpoints/sam_loveda'),
        filename='{epoch}-{val_mean_iou:.4f}',
        save_top_k=3,
        monitor='val_mean_iou',
        mode='max',
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)