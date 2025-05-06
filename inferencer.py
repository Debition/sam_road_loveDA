import numpy as np
import os
import imageio
import torch
import cv2

from utils import load_config, create_output_dir_and_save_config
from dataset import cityscale_data_partition, read_rgb_img
from dataset import spacenet_data_partition
from model import SAMRoad
import graph_extraction
import graph_utils
import triage
import pickle
import scipy
import rtree
from collections import defaultdict
import time
import random
import glob

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    "--config", default=None, help="model config."
)
parser.add_argument(
    "--output_dir", default=None, help="Name of the output dir, if not specified will use timestamp"
)
parser.add_argument("--device", default="cuda", help="device to use for training")
args = parser.parse_args()


def get_all_image_indices(dataset_type):
    """获取数据集中的所有图像索引"""
    if dataset_type == 'cityscale':
        # 获取所有图像文件
        image_files = glob.glob('./cityscale/20cities/region_*_sat.png')
        # 从文件名中提取索引
        indices = [int(os.path.basename(f).split('_')[1]) for f in image_files]
        return sorted(indices)
    elif dataset_type == 'spacenet':
        # 获取所有图像文件
        image_files = glob.glob('./spacenet/RGB_1.0_meter/*__rgb.png')
        # 从文件名中提取索引
        indices = [os.path.basename(f).split('__')[0] for f in image_files]
        return sorted(indices)
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")


def crop_random_patch(img, patch_size=512):
    """从大图中随机裁剪一个patch_size x patch_size的小图"""
    h, w = img.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(f"输入图像尺寸({h}x{w})小于patch_size({patch_size})")
    
    # 随机选择裁剪的起始位置
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    
    # 裁剪图像
    patch = img[y:y+patch_size, x:x+patch_size]
    return patch, (x, y)


def infer_one_img(net, img, config):
    """处理单张512x512的图像"""
    # 确保输入图像是512x512
    if img.shape[0] != 512 or img.shape[1] != 512:
        raise ValueError(f"输入图像尺寸必须是512x512，当前尺寸为{img.shape}")
    
    # 转换为tensor并添加batch维度
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(args.device)
    
    with torch.no_grad():
        # 预测掩码和图像特征
        mask_scores, img_features = net.infer_masks_and_img_features(img_tensor)
    
    # 获取预测的掩码
    keypoint_mask = mask_scores[0, :, :, 0].cpu().numpy()
    road_mask = mask_scores[0, :, :, 1].cpu().numpy()
    
    # 转换为0-255的uint8格式
    keypoint_mask = (keypoint_mask * 255).astype(np.uint8)
    road_mask = (road_mask * 255).astype(np.uint8)
    
    # 提取图结构
    graph_points = graph_extraction.extract_graph_points(keypoint_mask, road_mask, config)
    if graph_points.shape[0] == 0:
        return graph_points, np.zeros((0, 2), dtype=np.int32), keypoint_mask, road_mask
    
    # 构建R树用于空间查询
    graph_rtree = rtree.index.Index()
    for i, v in enumerate(graph_points):
        x, y = v
        graph_rtree.insert(i, (x, y, x, y))
    
    # 准备拓扑关系预测的数据
    patch_points = graph_points
    patch_kdtree = scipy.spatial.KDTree(patch_points)
    
    # 查询每个点的邻居
    knn_d, knn_idx = patch_kdtree.query(patch_points, k=config.MAX_NEIGHBOR_QUERIES + 1, 
                                       distance_upper_bound=config.NEIGHBOR_RADIUS)
    knn_idx = knn_idx[:, 1:]  # 移除自身
    
    # 准备点对数据
    src_idx = np.tile(np.arange(len(patch_points))[:, np.newaxis], (1, config.MAX_NEIGHBOR_QUERIES))
    valid = knn_idx < len(patch_points)
    tgt_idx = np.where(valid, knn_idx, src_idx)
    pairs = np.stack([src_idx, tgt_idx], axis=-1)
    
    # 转换为tensor
    batch_points = torch.tensor(patch_points, device=args.device).unsqueeze(0)
    batch_pairs = torch.tensor(pairs, device=args.device).unsqueeze(0)
    batch_valid = torch.tensor(valid, device=args.device).unsqueeze(0)
    
    # 预测拓扑关系
    with torch.no_grad():
        topo_scores = net.infer_toponet(img_features, batch_points, batch_pairs, batch_valid)
    
    # 处理拓扑关系预测结果
    topo_scores = torch.where(torch.isnan(topo_scores), -100.0, topo_scores).squeeze(-1).cpu().numpy()
    
    # 聚合边分数
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    
    for si in range(len(patch_points)):
        for pi in range(config.MAX_NEIGHBOR_QUERIES):
            if not valid[si, pi]:
                continue
            src_idx, tgt_idx = pairs[si, pi]
            edge_score = topo_scores[0, si, pi]
            edge_scores[(src_idx, tgt_idx)] += edge_score
            edge_counts[(src_idx, tgt_idx)] += 1.0
    
    # 平均边分数并过滤
    pred_edges = []
    for edge, score_sum in edge_scores.items():
        score = score_sum / edge_counts[edge]
        if score > config.TOPO_THRESHOLD:
            pred_edges.append(edge)
    
    pred_edges = np.array(pred_edges).reshape(-1, 2)
    pred_nodes = graph_points[:, ::-1]  # 转换为rc坐标
    
    return pred_nodes, pred_edges, keypoint_mask, road_mask


if __name__ == "__main__":
    config = load_config(args.config)
    
    # 构建评估模型
    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    net = SAMRoad(config)

    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(f'##### Loading Trained CKPT {args.checkpoint} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)

    if config.DATASET == 'cityscale':
        rgb_pattern = './cityscale/20cities/region_{}_sat.png'
        gt_graph_pattern = 'cityscale/20cities/region_{}_graph_gt.pickle'
    elif config.DATASET == 'spacenet':
        rgb_pattern = './spacenet/RGB_1.0_meter/{}__rgb.png'
        gt_graph_pattern = './spacenet/RGB_1.0_meter/{}__gt_graph.p'
    
    # 获取所有图像索引
    all_img_indices = get_all_image_indices(config.DATASET)
    print(f"找到 {len(all_img_indices)} 张图像")
    
    output_dir_prefix = './save/infer_'
    if args.output_dir:
        output_dir = create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=f'./save/{args.output_dir}')
    else:
        output_dir = create_output_dir_and_save_config(output_dir_prefix, config)
    
    # 创建保存裁切图像的目录
    patches_save_dir = os.path.join(output_dir, 'patches')
    if not os.path.exists(patches_save_dir):
        os.makedirs(patches_save_dir)
    
    total_inference_seconds = 0.0

    for img_id in all_img_indices:
        print(f'Processing {img_id}')
        # 读取原始大图
        img = read_rgb_img(rgb_pattern.format(img_id))
        
        try:
            # 随机裁剪512x512的小图
            img_patch, (x, y) = crop_random_patch(img, patch_size=512)
            
            # 保存裁切的图像
            patch_save_path = os.path.join(patches_save_dir, f'{img_id}_patch.png')
            cv2.imwrite(patch_save_path, cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
            
            # 保存裁切位置信息
            patch_info = {
                'img_id': img_id,
                'crop_position': (x, y),
                'patch_size': 512
            }
            patch_info_path = os.path.join(patches_save_dir, f'{img_id}_patch_info.p')
            with open(patch_info_path, 'wb') as f:
                pickle.dump(patch_info, f)
            
            start_seconds = time.time()
            # 推理
            pred_nodes, pred_edges, itsc_mask, road_mask = infer_one_img(net, img_patch, config)
            end_seconds = time.time()
            total_inference_seconds += (end_seconds - start_seconds)

            # 保存掩码结果
            mask_save_dir = os.path.join(output_dir, 'mask')
            if not os.path.exists(mask_save_dir):
                os.makedirs(mask_save_dir)
            cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_road.png'), road_mask)
            cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_itsc.png'), itsc_mask)

            # 保存可视化结果
            viz_save_dir = os.path.join(output_dir, 'viz')
            if not os.path.exists(viz_save_dir):
                os.makedirs(viz_save_dir)
            viz_img = triage.visualize_image_and_graph(img_patch, pred_nodes / 512, pred_edges, 512)
            cv2.imwrite(os.path.join(viz_save_dir, f'{img_id}.png'), viz_img)

            # 保存图结构
            graph_save_dir = os.path.join(output_dir, 'graph')
            if not os.path.exists(graph_save_dir):
                os.makedirs(graph_save_dir)
            graph_save_path = os.path.join(graph_save_dir, f'{img_id}.p')
            with open(graph_save_path, 'wb') as file:
                pickle.dump({
                    'nodes': pred_nodes,
                    'edges': pred_edges,
                    'crop_position': (x, y)
                }, file)
            
            print(f'Done for {img_id}.')
            
        except ValueError as e:
            print(f'Error processing {img_id}: {str(e)}')
            continue
    
    # 记录推理时间
    time_txt = f'Inference completed for {args.config} in {total_inference_seconds} seconds.'
    print(time_txt)
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w') as f:
        f.write(time_txt)