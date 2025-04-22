import os
import numpy as np
import matplotlib.pyplot as plt
import random
from loveda_dataset import LoveDADataset

# 设置随机种子以便结果可复现（可选）
random.seed(42)

# 数据集路径
root_dir = 'training_data'

# 只加载城市地区的数据
urban_dataset = LoveDADataset(root_dir, split='Train', regions=['Urban'])
print(f"加载了 {len(urban_dataset)} 个城市图像样本")

def show_top_road_images(dataset, top_n=10):
    """
    显示道路像素占比最高的几个图像，全部放在一张大图中
    
    参数:
    dataset: 数据集
    top_n: 显示前几个道路占比最高的图像
    """
    if len(dataset) == 0:
        print("数据集为空！")
        return
    
    # 计算每个图像的道路像素占比
    road_percentages = []
    
    print("计算每个图像的道路像素占比...")
    for i in range(len(dataset)):
        sample = dataset[i]
        road_mask = sample['road_mask'].numpy()
        total_pixels = road_mask.size
        road_pixels = np.sum(road_mask > 0.5)
        percentage = 100.0 * road_pixels / total_pixels
        road_percentages.append((i, percentage, sample['sample_id']))
    
    # 按道路像素占比从高到低排序
    road_percentages.sort(key=lambda x: x[1], reverse=True)
    
    # 显示排序结果
    print("\n道路像素占比排名 (前20名):")
    for i, (idx, percentage, sample_id) in enumerate(road_percentages[:20]):
        print(f"第{i+1}名: 样本 {sample_id}, 道路占比: {percentage:.2f}%")
    
    # 显示top_n个图像，全部放在一张大图中
    top_n = min(top_n, len(road_percentages))
    
    # 计算合适的行列数
    if top_n <= 3:
        rows, cols = 1, top_n
    else:
        cols = min(5, top_n)  # 每行最多5个图像
        rows = (top_n + cols - 1) // cols  # 向上取整
    
    # 创建一个大图，包含原图和道路掩码
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # 确保axs是二维数组
    if rows == 1 and cols == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = axs.reshape(1, -1)
    elif cols == 1:
        axs = axs.reshape(-1, 1)
    
    # 遍历前top_n个图像
    for i in range(top_n):
        idx, percentage, sample_id = road_percentages[i]
        sample = dataset[idx]
        
        # 获取图像和道路掩码
        img = sample['image'].permute(1, 2, 0).numpy()
        road_mask = sample['road_mask'].numpy()
        
        # 计算当前子图的位置
        row, col = i // cols, i % cols
        
        # 创建道路掩码叠加图
        road_overlay = img.copy()
        yellow_mask = np.zeros_like(img)
        yellow_mask[road_mask > 0.5] = [1.0, 1.0, 0.0]  # 黄色
        
        alpha = 0.7
        road_overlay = img * (1 - alpha * road_mask[:,:,np.newaxis]) + yellow_mask * alpha * road_mask[:,:,np.newaxis]
        
        # 在相应位置显示叠加后的图像
        axs[row, col].imshow(road_overlay)
        axs[row, col].set_title(f"#{i+1} - ID:{sample_id}\n道路占比: {percentage:.2f}%")
        axs[row, col].axis('off')  # 关闭坐标轴
    
    # 隐藏未使用的子图
    for i in range(top_n, rows * cols):
        row, col = i // cols, i % cols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)  # 调整子图之间的间距
    plt.suptitle(f"道路像素占比最高的{top_n}个图像", fontsize=16, y=0.98)
    plt.show()

# 显示道路像素占比最高的城市图像
print(f"\n显示道路像素占比最高的12个城市图像...")
if len(urban_dataset) > 0:
    show_top_road_images(urban_dataset, top_n=12)

# 也可以看看农村地区的道路情况
rural_dataset = LoveDADataset(root_dir, split='Train', regions=['Rural'])
print(f"\n加载了 {len(rural_dataset)} 个农村图像样本")

# 显示农村地区道路像素占比最高的图像
if len(rural_dataset) > 0:
    print(f"\n显示道路像素占比最高的10个农村图像...")
    show_top_road_images(rural_dataset, top_n=10) 