import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import argparse
from pathlib import Path

# 直接导入SAMLoveDA，不使用sam_road包前缀
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 添加当前目录到导入路径
from sam_loveda import SAMLoveDA  # 直接导入

# 颜色映射（与训练代码保持一致）
COLOR_MAP = [
    [0, 0, 0],         # 0: 背景
    [255, 0, 0],       # 1: 建筑
    [255, 255, 0],     # 2: 道路
    [0, 0, 255],       # 3: 水体
    [159, 129, 183],   # 4: 草地
    [0, 255, 0],       # 5: 森林
    [255, 195, 128],   # 6: 农田
    [128, 128, 128]    # 7: 其他
]

CLASS_NAMES = ["other","background", "building", "road", "water", "grass", "forest", "farmland"]

def load_config():
    """加载与训练时相同的配置"""
    class DictConfig(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self
            
    config = DictConfig()
    
    # 需要与训练时使用的配置一致
    config['SAM_VERSION'] = 'vit_b'
    config['PATCH_SIZE'] = 512  # 确认与训练时一致
    config['USE_SAM_DECODER'] = False
    config['NUM_CLASSES'] = 8
    config['FOCAL_LOSS'] = True
    config['FOCAL_ALPHA'] = 0.25
    config['FOCAL_GAMMA'] = 2.0
    config['ENCODER_LORA'] = True
    config['LORA_RANK'] = 4
    config['FREEZE_ENCODER'] = False
    
    return config

def preprocess_image(image_path, target_size=512):
    """预处理输入图像"""
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    
    # 调整图像大小
    img = img.resize((target_size, target_size), Image.BILINEAR)
    
    # 转换为numpy数组
    img_np = np.array(img) / 255.0  # 归一化到[0,1]
    
    # 转换为tensor并添加批次维度
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, img_np

def predict(model, image_tensor, device):
    """使用模型进行预测"""
    model.eval()
    with torch.no_grad():
        # 将图像移到正确的设备
        image_tensor = image_tensor.to(device)
        
        # 进行推理
        class_pred, mask_scores = model.infer_masks(image_tensor)
        
        # 将结果移回CPU
        class_pred = class_pred.cpu().numpy()
        mask_scores = mask_scores.cpu().numpy()
    
    return class_pred[0], mask_scores[0]  # 返回第一个样本的结果（去掉批次维度）

def create_color_mask(pred_mask):
    """创建彩色掩码图像"""
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_idx, color in enumerate(COLOR_MAP):
        color_mask[pred_mask == cls_idx] = color
        
    return color_mask

def visualize_result(original_img, pred_mask, save_path=None, background_class=1):
    """可视化预测结果，降低背景类的透明度"""
    # 创建彩色掩码
    color_mask = create_color_mask(pred_mask)
    
    # 创建图形
    plt.figure(figsize=(16, 8))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    # 预测掩码（彩色）
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(color_mask)
    plt.axis('off')
    
    # 融合叠加 - 背景类较低透明度，前景类较高透明度
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    
    # 创建初始叠加图像
    overlay = original_img.copy()
    
    # 前景和背景的遮罩
    bg_mask = (pred_mask == background_class)
    fg_mask = ~bg_mask
    
    # 背景使用低透明度 (0.1)
    if np.any(bg_mask):
        overlay[bg_mask] = original_img[bg_mask] * 0.9 + color_mask[bg_mask] / 255.0 * 0.1
    
    # 前景使用高透明度 (0.5)
    if np.any(fg_mask):
        overlay[fg_mask] = original_img[fg_mask] * 0.5 + color_mask[fg_mask] / 255.0 * 0.5
    
    plt.imshow(overlay)
    plt.axis('off')
    
    # 添加图例
    handles = []
    for i, name in enumerate(CLASS_NAMES):
        color = np.array(COLOR_MAP[i]) / 255.0
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))
    
    plt.figlegend(handles, CLASS_NAMES, loc='lower center', ncol=len(CLASS_NAMES), bbox_to_anchor=(0.5, 0))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        print(f"结果已保存到: {save_path}")
    
    plt.show()
    
    # 打印类别统计
    unique_classes, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size
    
    print("\n预测类别分布:")
    for cls, count in zip(unique_classes, counts):
        print(f"  {CLASS_NAMES[cls]}: {count} 像素 ({count/total_pixels*100:.2f}%)")

def save_results(pred_mask, class_scores, save_dir, filename):
    """保存预测结果"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # 保存类别掩码
    color_mask = create_color_mask(pred_mask)
    color_mask_path = os.path.join(save_dir, f"{base_name}_mask_color.png")
    Image.fromarray(color_mask).save(color_mask_path)
    
    # 保存类别索引掩码
    mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
    Image.fromarray(pred_mask.astype(np.uint8)).save(mask_path)
    
    # 保存置信度图（每个类别的分数图）
    for i in range(class_scores.shape[0]):
        score_map = (class_scores[i] * 255).astype(np.uint8)
        score_path = os.path.join(save_dir, f"{base_name}_score_class{i}.png")
        Image.fromarray(score_map).save(score_path)
    
    print(f"分类掩码已保存到: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='土地分类推理程序')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/epoch=9-val_mean_iou=0.2686.ckpt', 
                        help='模型检查点路径')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='使用设备 (cuda/cpu)')
    parser.add_argument('--bg-class', type=int, default=1, help='背景类的索引 (默认: 1)')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置和模型
    config = load_config()
    model = SAMLoveDA(config)
    
    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    # 预处理图像
    image_tensor, original_img = preprocess_image(args.image, target_size=config['PATCH_SIZE'])
    
    # 进行预测
    print("执行推理...")
    pred_mask, class_scores = predict(model, image_tensor, device)
    
    # 可视化结果
    save_path = os.path.join(args.output, f"{Path(args.image).stem}_visualization.png")
    visualize_result(original_img, pred_mask, save_path=save_path, background_class=args.bg_class)
    
    # 保存结果
    save_results(pred_mask, class_scores, args.output, args.image)
    
    print("推理完成！")

if __name__ == "__main__":
    main()
