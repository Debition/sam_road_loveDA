from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from datetime import timedelta
import os

from utils import load_config
from loveda_dataset import create_loveda_dataloaders
from sam_loveda import SAMLoveDA

import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


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
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 打印训练配置摘要
    print("\n" + "="*50)
    print("SAM-LoveDA 训练开始")
    print("="*50)
    print(f"配置文件: {args.config}")
    print(f"模型版本: {config.SAM_VERSION}")
    print(f"图像尺寸: {config.PATCH_SIZE}x{config.PATCH_SIZE}")
    print(f"批次大小: {config.TRAIN_BATCH_SIZE} (训练) / {config.VAL_BATCH_SIZE} (验证)")
    print(f"训练周期: {config.TRAIN_EPOCHS}")
    print(f"学习率: {config.BASE_LR} (调度器: {config.LR_SCHEDULER})")
    print(f"使用SAM解码器: {config.USE_SAM_DECODER}")
    print(f"使用LoRA: {config.ENCODER_LORA} (秩: {config.LORA_RANK})")
    print(f"冻结编码器: {config.FREEZE_ENCODER}")
    print(f"数据集: {config.DATASET_ROOT} (区域: {config.REGIONS})")
    print(f"精度: {args.precision}")
    print(f"开发模式: {dev_run}")
    print("="*50 + "\n")

    # 打印GPU使用情况
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"可用GPU数量: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            print(f"GPU {i}: {gpu_name} - 显存: {total_memory:.2f} GB")
    else:
        print("警告: 未检测到可用的GPU")
    print()

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
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    # 创建输出目录
    checkpoint_dir = config.get('CHECKPOINT_DIR', 'checkpoints/sam_loveda')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 设置回调
    callbacks = []
    
    # 模型保存回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_mean_iou:.4f}',
        save_top_k=3,
        monitor='val_mean_iou',
        mode='max',
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # 进度条回调
    progress_bar = TQDMProgressBar(refresh_rate=1)
    callbacks.append(progress_bar)
    
    # 早停回调
    if not dev_run:
        early_stop_callback = EarlyStopping(
            monitor='val_mean_iou',
            patience=10,
            verbose=True,
            mode='max'
        )
        callbacks.append(early_stop_callback)

    # WandB日志记录器
    wandb_logger = WandbLogger()

    print(f"数据加载完成 - 训练集: {len(train_loader.dataset)}个样本, 批次数: {len(train_loader)}")
    print(f"           - 验证集: {len(val_loader.dataset)}个样本, 批次数: {len(val_loader)}")
    print(f"开始训练过程...\n")

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        log_every_n_steps=config.get('LOG_INTERVAL', 50),
        )

    try:
        # 训练模型
        trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader, 
                    ckpt_path=args.resume if args.resume else None)
        
        # 训练结束后打印摘要
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*50)
        print("训练完成!")
        print(f"总训练时间: {timedelta(seconds=int(elapsed_time))}")
        print(f"保存路径: {checkpoint_callback.dirpath}")
        print(f"最佳模型: {checkpoint_callback.best_model_path} (IoU: {checkpoint_callback.best_model_score:.4f})")
        print(f"WandB日志: {wandb_logger.experiment.url}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n训练被用户中断!")
        elapsed_time = time.time() - start_time
        print(f"已训练时间: {timedelta(seconds=int(elapsed_time))}")
        if os.path.exists(os.path.join(checkpoint_dir, 'last.ckpt')):
            print(f"可以从最后检查点恢复: {os.path.join(checkpoint_dir, 'last.ckpt')}")