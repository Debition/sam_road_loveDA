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

def threshold_prediction(mask_scores, base_threshold=0.2, high_threshold=0.5, background_class=1):
    """
    使用阈值决策而非简单argmax
    对于各个类别，如果分数超过阈值，则分配该类别
    如果多个类别超过阈值，取分数最高的类别
    """
    h, w = mask_scores.shape[1], mask_scores.shape[2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 设置类别特定阈值
    thresholds = np.ones(mask_scores.shape[0]) * base_threshold
    thresholds[background_class] = high_threshold  # 背景类需要更高阈值
    
    # 对每个像素位置使用阈值决策
    for i in range(h):
        for j in range(w):
            # 找出超过阈值的类别
            valid_classes = []
            for c in range(mask_scores.shape[0]):
                if mask_scores[c, i, j] > thresholds[c]:
                    valid_classes.append((c, mask_scores[c, i, j]))
            
            # 如果有超过阈值的类别，选择置信度最高的
            if valid_classes:
                pred_mask[i, j] = max(valid_classes, key=lambda x: x[1])[0]
            else:
                # 如果没有类别超过阈值，默认为背景
                pred_mask[i, j] = background_class
    
    return pred_mask

def modify_class_scores(mask_scores, boost_road=1.5, boost_water=2.0, boost_barren=2.0, boost_forest=1.2):
    """提升少数类别的分数"""
    modified_scores = mask_scores.copy()
    
    # 增强特定类别的置信度（索引：3=道路，4=水体，5=贫瘠地，6=森林）
    modified_scores[3] *= boost_road
    modified_scores[4] *= boost_water
    modified_scores[5] *= boost_barren
    modified_scores[6] *= boost_forest
    
    return modified_scores

def visualize_score_maps(original_img, pred_mask, class_scores, save_path=None):
    """可视化所有类别的分数图"""
    num_classes = class_scores.shape[0]
    
    # 计算图表大小 - 原图+预测掩码+所有类别分数图
    grid_size = int(np.ceil(np.sqrt(num_classes + 2)))
    
    # 创建一个更大的图
    plt.figure(figsize=(grid_size * 4, grid_size * 4))
    
    # 原始图像
    plt.subplot(grid_size, grid_size, 1)
    plt.title("原始图像")
    plt.imshow(original_img)
    plt.axis('off')
    
    # 预测掩码（彩色）
    plt.subplot(grid_size, grid_size, 2)
    plt.title("预测掩码")
    color_mask = create_color_mask(pred_mask)
    plt.imshow(color_mask)
    plt.axis('off')
    
    # 各个类别的分数图
    for i in range(num_classes):
        plt.subplot(grid_size, grid_size, i + 3)
        plt.title(f"类别 {i}: {CLASS_NAMES[i]}")
        
        # 使用热力图显示分数
        plt.imshow(class_scores[i], cmap='hot', vmin=0, vmax=1)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        print(f"分数图可视化已保存到: {save_path}")
    
    plt.show()

def load_model(config, checkpoint_path, device):
    """加载模型和检查点"""
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = SAMLoveDA(config).to(device)
    
    # 加载检查点
    if checkpoint_path:
        print(f"加载检查点: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 使用非严格模式加载，允许部分参数不匹配
            model.load_state_dict(state_dict, strict=False)
            print("检查点加载成功")
        except Exception as e:
            print(f"警告: 加载检查点时出现问题: {str(e)}")
            print("继续使用未完全加载的模型...")
    
    model.eval()
    return model

def visualize_batch_results(original_imgs, pred_masks, title, save_path=None, gt_masks=None):
    """将一批图像的结果合并显示，训练集包含GT掩码"""
    n_samples = len(original_imgs)
    n_cols = 4 if gt_masks is not None else 3
    fig = plt.figure(figsize=(5*n_cols, 4*n_samples))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    for idx in range(n_samples):
        # 原始图像
        plt.subplot(n_samples, n_cols, idx*n_cols + 1)
        if idx == 0:
            plt.title("Original Image")
        plt.imshow(original_imgs[idx])
        plt.axis('off')
        
        # GT掩码（如果有）
        if gt_masks is not None:
            plt.subplot(n_samples, n_cols, idx*n_cols + 2)
            if idx == 0:
                plt.title("Ground Truth")
            gt_color_mask = create_color_mask(gt_masks[idx])
            plt.imshow(gt_color_mask)
            plt.axis('off')
        
        # 预测掩码
        plt.subplot(n_samples, n_cols, idx*n_cols + (3 if gt_masks is not None else 2))
        if idx == 0:
            plt.title("Predicted Mask")
        color_mask = create_color_mask(pred_masks[idx])
        plt.imshow(color_mask)
        plt.axis('off')
        
        # 叠加结果
        plt.subplot(n_samples, n_cols, idx*n_cols + (4 if gt_masks is not None else 3))
        if idx == 0:
            plt.title("Overlay")
        overlay = original_imgs[idx].copy()
        
        # 前景和背景的遮罩
        bg_mask = (pred_masks[idx] == 1)  # 假设1是背景类
        fg_mask = ~bg_mask
        
        # 背景使用低透明度 (0.1)
        if np.any(bg_mask):
            overlay[bg_mask] = original_imgs[idx][bg_mask] * 0.9 + color_mask[bg_mask] / 255.0 * 0.1
        
        # 前景使用高透明度 (0.5)
        if np.any(fg_mask):
            overlay[fg_mask] = original_imgs[idx][fg_mask] * 0.5 + color_mask[fg_mask] / 255.0 * 0.5
        
        plt.imshow(overlay)
        plt.axis('off')
    
    # 添加图例
    handles = []
    for i, name in enumerate(CLASS_NAMES):
        color = np.array(COLOR_MAP[i]) / 255.0
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))
    
    fig.legend(handles, CLASS_NAMES, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    plt.tight_layout()
    
    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"批次结果已保存到: {save_path}")
    
    plt.show()
    plt.close()

def process_dataset_samples(model, device, data_root, dataset_type, area_type, num_samples, args):
    """处理指定数据集中的样本"""
    # 构建图像目录路径
    image_dir = os.path.join(data_root, dataset_type, area_type, 'images_png')
    mask_dir = os.path.join(data_root, dataset_type, area_type, 'masks_png')
    
    # 获取所有PNG图像文件
    all_images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    # 随机采样
    if len(all_images) > num_samples:
        selected_images = np.random.choice(all_images, num_samples, replace=False)
    else:
        selected_images = all_images
    
    print(f"\n处理 {dataset_type}/{area_type} 数据集中的 {len(selected_images)} 个样本:")
    print("=" * 60)
    
    # 存储所有样本的指标和结果
    metrics = {
        'class_ious': np.zeros((len(selected_images), len(CLASS_NAMES))),
        'mean_ious': []
    }
    
    batch_results = {
        'original_imgs': [],
        'pred_masks': [],
        'gt_masks': [] if dataset_type.lower() == 'train' else None,
        'class_scores': []
    }
    
    # 处理每个选定的图像
    for idx, img_name in enumerate(selected_images):
        img_path = os.path.join(image_dir, img_name)
        print(f"\n处理图像 {idx+1}/{len(selected_images)}: {img_name}")
        
        # 预处理图像
        image_tensor, original_img = preprocess_image(img_path, target_size=model.config['PATCH_SIZE'])
        
        # 如果是训练集，加载GT掩码
        if dataset_type.lower() == 'train':
            mask_path = os.path.join(mask_dir, img_name)
            if os.path.exists(mask_path):
                gt_mask = Image.open(mask_path)
                gt_mask = gt_mask.resize((model.config['PATCH_SIZE'], model.config['PATCH_SIZE']), Image.NEAREST)
                gt_mask = np.array(gt_mask)
                batch_results['gt_masks'].append(gt_mask)
            else:
                print(f"警告: 未找到掩码文件 {mask_path}")
                batch_results['gt_masks'].append(np.zeros_like(original_img[:,:,0]))
        
        # 进行预测
        _, class_scores = predict(model, image_tensor, device)
        
        # 增强少数类别分数
        class_scores = modify_class_scores(
            class_scores,
            boost_road=args.boost_road,
            boost_water=args.boost_water,
            boost_barren=args.boost_barren,
            boost_forest=args.boost_forest
        )
        
        # 使用阈值法替代argmax
        pred_mask = threshold_prediction(
            class_scores,
            base_threshold=args.base_threshold,
            high_threshold=args.bg_threshold,
            background_class=args.bg_class
        )
        
        # 存储结果用于批量可视化
        batch_results['original_imgs'].append(original_img)
        batch_results['pred_masks'].append(pred_mask)
        batch_results['class_scores'].append(class_scores)
        
        # 计算每个类别的像素数量
        unique_classes, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size
        
        print("\n类别分布:")
        for cls, count in zip(unique_classes, counts):
            percentage = count/total_pixels*100
            print(f"  {CLASS_NAMES[cls]}: {count} 像素 ({percentage:.2f}%)")
            metrics['class_ious'][idx, cls] = percentage
    
    # 批量可视化结果
    save_dir = os.path.join(args.output, dataset_type, area_type)
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成批次标题
    batch_title = f"{area_type} {dataset_type} Results"
    save_path = os.path.join(save_dir, f"batch_results_{area_type.lower()}_{dataset_type.lower()}.png")
    
    # 显示批次结果
    visualize_batch_results(
        batch_results['original_imgs'],
        batch_results['pred_masks'],
        batch_title,
        save_path=save_path,
        gt_masks=batch_results['gt_masks']
    )
    
    # 保存详细结果
    for idx, img_name in enumerate(selected_images):
        save_results(
            batch_results['pred_masks'][idx],
            batch_results['class_scores'][idx],
            save_dir,
            img_name
        )
    
    # 计算平均指标
    mean_class_distribution = np.mean(metrics['class_ious'], axis=0)
    print("\n平均类别分布:")
    for cls_idx, mean_percentage in enumerate(mean_class_distribution):
        print(f"  {CLASS_NAMES[cls_idx]}: {mean_percentage:.2f}%")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='土地分类推理程序')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='使用设备 (cuda/cpu)')
    parser.add_argument('--bg-class', type=int, default=1, help='背景类的索引 (默认: 1)')
    parser.add_argument('--boost-road', type=float, default=8, help='道路类别分数提升倍数')
    parser.add_argument('--boost-water', type=float, default=1, help='水体类别分数提升倍数')
    parser.add_argument('--boost-barren', type=float, default=1, help='贫瘠地类别分数提升倍数')
    parser.add_argument('--boost-forest', type=float, default=2, help='森林类别分数提升倍数')
    parser.add_argument('--use-threshold', action='store_true', help='使用阈值决策代替argmax')
    parser.add_argument('--base-threshold', type=float, default=0.15, help='基本阈值')
    parser.add_argument('--bg-threshold', type=float, default=0.3, help='背景类阈值')
    parser.add_argument('--data-root', type=str, default='dataset', help='数据集根目录')
    parser.add_argument('--num-samples', type=int, default=5, help='每个子集要处理的样本数量')
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置和模型
    config = load_config()
    model = load_model(config, args.checkpoint, device)
    
    # 处理训练集
    print("\n处理训练集:")
    print("=" * 80)
    train_urban_metrics = process_dataset_samples(model, device, args.data_root, 'Train', 'Urban', args.num_samples, args)
    train_rural_metrics = process_dataset_samples(model, device, args.data_root, 'Train', 'Rural', args.num_samples, args)
    
    # 处理测试集
    print("\n处理测试集:")
    print("=" * 80)
    test_urban_metrics = process_dataset_samples(model, device, args.data_root, 'Test', 'Urban', args.num_samples, args)
    test_rural_metrics = process_dataset_samples(model, device, args.data_root, 'Test', 'Rural', args.num_samples, args)
    
    # 打印总结
    print("\n总体评估结果:")
    print("=" * 80)
    print("\n训练集结果:")
    print("-" * 40)
    print("城市区域平均类别分布:")
    for cls_idx, mean_percentage in enumerate(np.mean(train_urban_metrics['class_ious'], axis=0)):
        print(f"  {CLASS_NAMES[cls_idx]}: {mean_percentage:.2f}%")
    print("\n农村区域平均类别分布:")
    for cls_idx, mean_percentage in enumerate(np.mean(train_rural_metrics['class_ious'], axis=0)):
        print(f"  {CLASS_NAMES[cls_idx]}: {mean_percentage:.2f}%")
    
    print("\n测试集结果:")
    print("-" * 40)
    print("城市区域平均类别分布:")
    for cls_idx, mean_percentage in enumerate(np.mean(test_urban_metrics['class_ious'], axis=0)):
        print(f"  {CLASS_NAMES[cls_idx]}: {mean_percentage:.2f}%")
    print("\n农村区域平均类别分布:")
    for cls_idx, mean_percentage in enumerate(np.mean(test_rural_metrics['class_ious'], axis=0)):
        print(f"  {CLASS_NAMES[cls_idx]}: {mean_percentage:.2f}%")
    
    print("\n评估完成！")

if __name__ == "__main__":
    main()
