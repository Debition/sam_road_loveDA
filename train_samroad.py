from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config
from samroad_dataset import SatMapDataset
from sam_road import SAMRoad

import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

from samroad_dataset import DeepGlobeDataset

# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger('SAMRoad')

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return Config(config)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--precision', type=str, default='16-mixed', help='训练精度')
    parser.add_argument('--dev_run', action='store_true', help='是否进行开发运行')
    args = parser.parse_args()

    # 设置日志
    setup_logging()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 初始化模型
    model = SAMRoad(config)
    
    # 设置数据加载器
    train_dataset = DeepGlobeDataset(config, is_train=True)
    val_dataset = DeepGlobeDataset(config, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # 设置检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename='sam_road-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    # 设置学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 设置早停
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    # 设置训练器
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator='gpu',
        devices=1,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # 梯度累积
        val_check_interval=0.25,    # 更频繁的验证
        limit_train_batches=0.1 if args.dev_run else None,
        limit_val_batches=0.1 if args.dev_run else None,
        enable_progress_bar=True,
        logger=WandbLogger(project='sam_road', name=f'sam_road_{config.MODEL_VERSION}')
    )
    
    # 开始训练
    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume
        )
    except KeyboardInterrupt:
        logging.info("训练被用户中断")
    except Exception as e:
        logging.error(f"训练过程中发生错误: {str(e)}")
        raise e

if __name__ == '__main__':
    main()