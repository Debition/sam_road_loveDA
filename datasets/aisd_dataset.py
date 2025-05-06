import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import glob
import cv2
from collections import Counter
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)

class AISDDataset(Dataset):
    """
    AISD (Aerial Image Segmentation Dataset) 数据集加载器
    
    标签信息:
    - 建筑物: [255, 0, 0] -> 1
    - 道路: [0, 0, 255] -> 2
    - 背景: [255, 255, 255] -> 0
    
    城市RGB均值:
    - Berlin:  [79.94162, 84.72064, 78.94711]
    - Chicago: [86.46459, 85.73488, 77.14777]
    - Paris:   [82.46727, 92.82243, 88.05664]
    - Potsdam: [74.85480, 77.37761, 70.22035]
    - Tokyo:   [96.96883, 98.44344, 108.60135]
    - Zurich:  [62.36962, 66.11001, 60.32863]
    """
    def __init__(self, root_dir, split='train', cities=None, transform=None, patch_size=1024):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        
        # 标签颜色映射
        self.label_colors = {
            'background': [255, 255, 255],  # -> 0
            'building': [255, 0, 0],        # -> 1
            'road': [0, 0, 255]             # -> 2
        }
        
        # 创建颜色到类别的映射
        self.color_to_class = {}
        for i, (_, color) in enumerate(self.label_colors.items()):
            self.color_to_class[tuple(color)] = i
        
        if cities is None:
            self.cities = ['berlin', 'chicago', 'paris', 'potsdam', 'tokyo', 'zurich']
        else:
            self.cities = cities
            
        self.image_paths = []
        self.mask_paths = []
        self.cities_list = []  # 记录每个样本属于哪个城市
        
        # 收集所有图像和对应掩码的路径
        for city in self.cities:
            city_dir = os.path.join(root_dir, city)
            if not os.path.exists(city_dir):
                logger.warning(f"城市目录不存在: {city_dir}")
                continue
            
            # 获取所有图像文件
            img_files = glob.glob(os.path.join(city_dir, f"{city}*_image.png"))
            city_count = 0
            
            # 根据split划分数据集
            img_files.sort()  # 确保顺序一致
            num_files = len(img_files)
            if split == 'train':
                img_files = img_files[:int(num_files * 0.8)]  # 前80%用于训练
            elif split == 'val':
                img_files = img_files[int(num_files * 0.8):int(num_files * 0.9)]  # 中间10%用于验证
            else:  # test
                img_files = img_files[int(num_files * 0.9):]  # 最后10%用于测试
            
            for img_path in img_files:
                # 获取对应的标签文件路径
                img_filename = os.path.basename(img_path)
                mask_filename = img_filename.replace('_image.png', '_labels.png')
                mask_path = os.path.join(city_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.cities_list.append(city)
                    city_count += 1
                else:
                    logger.warning(f"找不到对应掩码文件: {mask_path}")
            
            logger.info(f"已加载 {city} 的 {city_count} 个{split}样本")
    
    def _convert_mask(self, mask):
        """将RGB掩码转换为类别索引"""
        h, w, c = mask.shape
        output = np.zeros((h, w), dtype=np.uint8)
        
        # 为每个类别创建掩码
        for class_idx, color in enumerate(self.label_colors.values()):
            mask_r = mask[:, :, 0] == color[0]
            mask_g = mask[:, :, 1] == color[1]
            mask_b = mask[:, :, 2] == color[2]
            class_mask = mask_r & mask_g & mask_b
            output[class_mask] = class_idx
            
        return output
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像和掩码
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        city = self.cities_list[idx]
        
        # 读取RGB图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取掩码（保持RGB格式以进行颜色匹配）
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # 获取样本ID信息
        img_filename = os.path.basename(img_path)
        sample_id = img_filename.replace('_image.png', '')
        
        # 根据配置调整图像大小
        if img.shape[0] != self.patch_size or img.shape[1] != self.patch_size:
            # 调整图像大小
            img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            # 调整掩码大小 - 使用最近邻插值以保持类别标签的整数性质
            mask = cv2.resize(mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        
        # 将RGB掩码转换为类别索引
        mask = self._convert_mask(mask)
        
        # 应用数据增强
        if self.transform:
            img, mask = self.transform(img, mask)
        
        # 转换为PyTorch张量
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0  # [C, H, W]
        mask = torch.from_numpy(mask).long()  # [H, W]
        
        # 确保维度正确
        assert img.shape[0] == 3, f"图像应该有3个通道，但得到了{img.shape[0]}个通道"
        assert img.shape[1] == img.shape[2] == self.patch_size, f"图像尺寸应该是{self.patch_size}x{self.patch_size}，但得到了{img.shape[1]}x{img.shape[2]}"
        assert mask.shape == (self.patch_size, self.patch_size), f"掩码尺寸应该是{self.patch_size}x{self.patch_size}，但得到了{mask.shape}"
        
        return {
            'image': img,
            'mask': mask,
            'sample_id': sample_id,
            'city': city,
            'img_path': img_path,
            'mask_path': mask_path
        }

class AISDTransform:
    """AISD数据集的数据增强类"""
    def __init__(self, split='train', config=None):
        self.split = split
        self.config = config if config is not None else {}
        
        # 基础颜色增强
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15
        )
        
    def __call__(self, img, mask):
        # 确保输入是连续的数组
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
        
        # 训练集应用增强
        if self.split == 'train' and self.config.get('AUGMENTATION', {}):
            aug_config = self.config.get('AUGMENTATION', {})
            
            # 随机翻转
            if aug_config.get('RANDOM_FLIP', True):
                if random.random() < 0.5:
                    img = np.fliplr(img)
                    mask = np.fliplr(mask)
                if random.random() < 0.5:
                    img = np.flipud(img)
                    mask = np.flipud(mask)
            
            # 随机旋转
            if aug_config.get('RANDOM_ROTATE', True):
                if random.random() < 0.5:
                    k = random.randint(1, 3)  # 90度的倍数
                    img = np.rot90(img, k=k)
                    mask = np.rot90(mask, k=k)
            
            # 随机缩放
            if aug_config.get('RANDOM_SCALE', None):
                if random.random() < 0.5:
                    scale_range = aug_config.get('RANDOM_SCALE')
                    scale = random.uniform(scale_range[0], scale_range[1])
                    h, w = img.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    
                    # 处理缩放后的尺寸
                    if scale > 1:  # 放大后需要裁剪
                        start_h = random.randint(0, new_h - h)
                        start_w = random.randint(0, new_w - w)
                        img = img[start_h:start_h+h, start_w:start_w+w]
                        mask = mask[start_h:start_h+h, start_w:start_w+w]
                    else:  # 缩小后需要填充
                        pad_h = (h - new_h) // 2
                        pad_w = (w - new_w) // 2
                        img = np.pad(img, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='reflect')
                        mask = np.pad(mask, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='constant', constant_values=0)
            
            # 颜色抖动
            if aug_config.get('COLOR_JITTER', True):
                if random.random() < 0.5:
                    img = torch.from_numpy(img.transpose(2, 0, 1))
                    img = self.color_jitter(img)
                    img = img.numpy().transpose(1, 2, 0)
        
        return img, mask

def create_aisd_dataloaders(config):
    """
    创建AISD数据集的数据加载器
    
    参数:
    config: 配置对象，包含数据集设置
    
    返回:
    tuple: (train_loader, val_loader, test_loader)
    """
    root_dir = config.DATASET_ROOT
    batch_size = config.TRAIN_BATCH_SIZE
    val_batch_size = config.VAL_BATCH_SIZE
    num_workers = config.NUM_WORKERS
    patch_size = config.PATCH_SIZE
    
    # 创建数据增强
    train_transform = AISDTransform(split='train', config=config)
    val_transform = AISDTransform(split='val', config=config)
    test_transform = AISDTransform(split='test', config=config)
    
    # 创建数据集
    train_dataset = AISDDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        patch_size=patch_size
    )
    
    val_dataset = AISDDataset(
        root_dir=root_dir,
        split='val',
        transform=val_transform,
        patch_size=patch_size
    )
    
    test_dataset = AISDDataset(
        root_dir=root_dir,
        split='test',
        transform=test_transform,
        patch_size=patch_size
    )
    
    # 打印数据集信息
    logger.info("\n" + "-"*20 + " 数据集信息 " + "-"*20)
    logger.info(f"数据集根目录: {root_dir}")
    logger.info(f"图像尺寸: {patch_size}x{patch_size}")
    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"验证集样本数: {len(val_dataset)}")
    logger.info(f"测试集样本数: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    logger.info(f"测试批次数: {len(test_loader)}")
    logger.info("-"*53 + "\n")
    
    return train_loader, val_loader, test_loader 