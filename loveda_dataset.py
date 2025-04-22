import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import os
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LoveDADataset(Dataset):
    """
    LoveDA数据集加载器
    类别标签:
    - 0: 无数据区域（应被忽略）
    - 1: 背景
    - 2: 建筑
    - 3: 道路
    - 4: 水体
    - 5: 荒地
    - 6: 森林
    - 7: 农田
    """
    def __init__(self, root_dir, split='Train', regions=None, transform=None, patch_size=1024):
        """
        参数:
        root_dir (str): 数据集根目录
        split (str): 数据集分割，可选 'Train', 'Val', 'Test'
        regions (list): 要包含的区域，可选 ['Urban', 'Rural'] 或其中之一
        transform (callable, optional): 应用于图像和掩码的可选变换
        patch_size (int): 图像尺寸，默认1024
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        
        if regions is None:
            self.regions = ['Urban', 'Rural']
        else:
            self.regions = regions
            
        self.image_paths = []
        self.mask_paths = []
        
        # 收集所有图像和对应掩码的路径
        for region in self.regions:
            img_dir = os.path.join(root_dir, split, region, 'images_png')
            mask_dir = os.path.join(root_dir, split, region, 'masks_png')
            
            img_files = glob.glob(os.path.join(img_dir, '*.png'))
            
            for img_path in img_files:
                img_filename = os.path.basename(img_path)
                mask_path = os.path.join(mask_dir, img_filename)
                
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像和掩码
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # 读取RGB图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 获取区域和样本ID信息
        img_filename = os.path.basename(img_path)
        sample_id = os.path.splitext(img_filename)[0]
        region = 'Urban' if '/Urban/' in img_path else 'Rural'
        
        # 根据配置调整图像大小
        orig_size = img.shape[0]  # 原始图像大小
        if orig_size != self.patch_size:
            # 调整图像大小
            img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            # 调整掩码大小 - 使用最近邻插值以保持类别标签的整数性质
            mask = cv2.resize(mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        
        # 创建道路掩码（类别3）
        road_mask = (mask == 3).astype(np.uint8) * 255
        
        # 创建建筑掩码（类别2）
        building_mask = (mask == 2).astype(np.uint8) * 255
        
        # 创建水体掩码（类别4）
        water_mask = (mask == 4).astype(np.uint8) * 255
        
        # 应用变换（如果有）
        if self.transform:
            img, mask, road_mask, building_mask, water_mask = self.transform(
                img, mask, road_mask, building_mask, water_mask)
        
        # 转换为PyTorch张量
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        road_mask = torch.from_numpy(road_mask).float() / 255.0
        building_mask = torch.from_numpy(building_mask).float() / 255.0
        water_mask = torch.from_numpy(water_mask).float() / 255.0
        
        return {
            'image': img,
            'mask': mask,
            'road_mask': road_mask,
            'building_mask': building_mask,
            'water_mask': water_mask,
            'sample_id': sample_id,
            'region': region,
            'img_path': img_path,
            'mask_path': mask_path
        }


def create_loveda_dataloaders(config):
    """
    创建LoveDA数据集的数据加载器
    
    参数:
    config (dict): 配置对象，包含数据集设置
    
    返回:
    tuple: 训练、验证和测试数据加载器的元组
    """
    root_dir = config.get('DATASET_ROOT', 'data/LoveDA')
    batch_size = config.get('TRAIN_BATCH_SIZE', 8)
    val_batch_size = config.get('VAL_BATCH_SIZE', batch_size)
    num_workers = config.get('NUM_WORKERS', 4) 
    regions = config.get('REGIONS', ['Urban', 'Rural'])
    patch_size = config.get('PATCH_SIZE', 1024)  # 获取图像尺寸
    
    # 从config中获取分割信息或使用默认值
    split_info = config.get('SPLIT_INFO', {})
    train_split = split_info.get('TRAIN', 'train')
    val_split = split_info.get('VAL', 'val')
    test_split = split_info.get('TEST', 'test')
    
    # 打印找到的数据路径信息(调试用)
    print(f"数据集根目录: {root_dir}")
    print(f"使用区域: {regions}")
    print(f"图像尺寸: {patch_size}x{patch_size}")
    print(f"训练分割: {train_split}, 验证分割: {val_split}, 测试分割: {test_split}")
    
    train_dataset = LoveDADataset(root_dir, split=train_split, regions=regions, patch_size=patch_size)
    val_dataset = LoveDADataset(root_dir, split=val_split, regions=regions, patch_size=patch_size)
    test_dataset = LoveDADataset(root_dir, split=test_split, regions=regions, patch_size=patch_size)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def visualize_sample(sample):
    """
    可视化数据集样本
    
    参数:
    sample (dict): 从数据集中获取的样本
    """
    img = sample['image'].permute(1, 2, 0).numpy()
    mask = sample['mask'].numpy()
    road_mask = sample['road_mask'].numpy()
    building_mask = sample['building_mask'].numpy()
    water_mask = sample['water_mask'].numpy()
    
    # 创建颜色标签可视化
    colors = [
        [0, 0, 0],       # 0: 无数据
        [128, 128, 128], # 1: 背景
        [255, 0, 0],     # 2: 建筑
        [255, 255, 0],   # 3: 道路
        [0, 0, 255],     # 4: 水体
        [159, 129, 183], # 5: 荒地
        [0, 255, 0],     # 6: 森林
        [255, 195, 128]  # 7: 农田
    ]
    
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        colored_mask[mask == i] = color
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    axs[0, 0].imshow(img)
    axs[0, 0].set_title(f"original image - {sample['region']} - {sample['sample_id']}")
    
    axs[0, 1].imshow(colored_mask)
    axs[0, 1].set_title('classification mask')
    
    axs[0, 2].imshow(road_mask, cmap='gray')
    axs[0, 2].set_title('road mask')
    
    axs[1, 0].imshow(building_mask, cmap='gray')
    axs[1, 0].set_title('building mask')
    
    axs[1, 1].imshow(water_mask, cmap='gray')
    axs[1, 1].set_title('water mask')
    
    # 空出一个位置
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例使用
    # 假设LoveDA数据集位于 'training_data' 文件夹下
    root_dir = 'training_data'
    
    # 创建数据集实例
    dataset = LoveDADataset(root_dir, split='Train')
    print(f"数据集大小: {len(dataset)}")
    
    # 获取并可视化一个样本
    sample = dataset[0]
    visualize_sample(sample)
    
    # 创建数据加载器
    dataloaders = create_loveda_dataloaders(root_dir, batch_size=4)
    
    # 打印数据加载器信息
    print(f"训练集大小: {len(dataloaders['train'].dataset)}")
    print(f"验证集大小: {len(dataloaders['val'].dataset)}")
    print(f"测试集大小: {len(dataloaders['test'].dataset)}") 