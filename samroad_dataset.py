import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import graph_utils
import rtree
import scipy
import pickle
import os
import addict
import json
import glob
import logging
import torch.nn.functional as F



def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def cityscale_data_partition():
    # dataset partition
    indrange_train = []
    indrange_test = []
    indrange_validation = []

    for x in range(180):
        if x % 10 < 8 :
            indrange_train.append(x)

        if x % 10 == 9:
            indrange_test.append(x)

        if x % 20 == 18:
            indrange_validation.append(x)

        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


def spacenet_data_partition():
    # dataset partition
    with open('./spacenet/data_split.json','r') as jf:
        data_list = json.load(jf)
        # data_list = data_list['test'] + data_list['validation'] + data_list['train']
    # train_list = [tile_index for _, tile_index in data_list['train']]
    # val_list = [tile_index for _, tile_index in data_list['validation']]
    # test_list = [tile_index for _, tile_index in data_list['test']]
    train_list = data_list['train']
    val_list = data_list['validation']
    test_list = data_list['test']
    return train_list, val_list, test_list


def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    patch_info = []
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info


class GraphLabelGenerator():
    def __init__(self, config, full_graph, coord_transform):
        self.config = config
        # full_graph: sat2graph format
        # coord_transform: lambda, [N, 2] array -> [N, 2] array
        # convert to igraph for high performance
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform)
        # find crossover points, we'll avoid predicting these as keypoints
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)
        # subdivide version
        # TODO: check proper resolution
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(self.full_graph_origin, self.subdivide_resolution)
        # np array, maybe faster
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])
        # pre-build spatial index
        # rtree for box queries
        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            # hack to insert single points
            self.graph_rtee.insert(i, (x, y, x, y))
        # kdtree for spherical query
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # pre-exclude points near crossover points
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # Find intersection points, these will always be kept in nms
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num, ), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0  # itsc points will always be kept

        # Points near crossover and intersections are interesting.
        # they will be more frequently sampled
        interesting_indices = set()
        interesting_radius = 32
        # near itsc
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(np.array(p), interesting_radius)
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num, ), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9
    
    def sample_patch(self, patch, rot_index = 0):
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices

        # Use NMS to downsample, params shall resemble inference time
        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            # print("==== Patch is empty ====")
            # this shall be rare, but if no points in side the patch, return null stuff
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = ([[0, 0]] * max_nbr_queries, [False] * max_nbr_queries, [False] * max_nbr_queries)
            return fake_points, [fake_sample] * sample_num

        patch_points = self.subdivide_points[patch_indices, :]
        
        # random scores to emulate different random configurations that all share a
        # similar spacing between sampled points
        # raise scores for intersction points so they are always kept
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        nms_radius = self.config.ROAD_NMS_RADIUS
        
        # kept_indces are into the patch_points array
        nmsed_points, kept_indices = graph_utils.nms_points(patch_points, nms_scores, radius=nms_radius, return_indices=True)
        # now this is into the subdivide graph
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]


        sample_num = self.config.TOPO_SAMPLE_NUM  # has to be greater than 1
        sample_weights = self.sample_weights[nmsed_indices]
        # indices into the nmsed points in the patch
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights))
        # indices into the subdivided graph
        sample_indices = nmsed_indices[sample_indices_in_nmsed]
        
        radius = self.config.NEIGHBOR_RADIUS
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES  # has to be greater than 1
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        # [n_sample, n_nbr]
        # k+1 because the nearest one is always self
        knn_d, knn_idx = nmsed_kdtree.query(sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius)


        samples = []

        for i in range(sample_num):
            source_node = sample_indices[i]
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]
            valid_nbr_indices = valid_nbr_indices[1:] # the nearest one is self so remove
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]  

            ### BFS to find immediate neighbors on graph
            reached_nodes = graph_utils.bfs_with_conditions(self.full_graph_subdivide, source_node, set(target_nodes), radius // self.subdivide_resolution)
            shall_connect = [t in reached_nodes for t in target_nodes]
            ###

            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)

            # zero-pad
            for i in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)

            samples.append((pairs, shall_connect, valid))

        # Transform points
        # [N, 2]
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        # homo for rot
        # [N, 3]
        nmsed_points = np.concatenate([nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1)
        trans = np.array([
            [1, 0, -0.5 * self.config.PATCH_SIZE],
            [0, 1, -0.5 * self.config.PATCH_SIZE],
            [0, 0, 1],
        ], dtype=np.float32)
        # ccw 90 deg in img (x, y)
        rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]
            
        # Add noise
        noise_scale = 1.0  # pixels
        nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)

        return nmsed_points, samples
    

def test_graph_label_generator():
    if not os.path.exists('debug'):
        os.mkdir('debug')

    dataset = 'spacenet'
    if dataset == 'cityscale':
        rgb_path = './cityscale/20cities/region_166_sat.png'
        # Load GT Graph
        gt_graph = pickle.load(open(f"./cityscale/20cities/region_166_refine_gt_graph.p",'rb'))
        coord_transform = lambda v : v[:, ::-1]
    elif dataset == 'spacenet':
        rgb_path = 'spacenet/RGB_1.0_meter/AOI_2_Vegas_210__rgb.png'
        # Load GT Graph
        gt_graph = pickle.load(open(f"spacenet/RGB_1.0_meter/AOI_2_Vegas_210__gt_graph.p",'rb'))
        # gt_graph = pickle.load(open(f"spacenet/RGB_1.0_meter/AOI_4_Shanghai_1061__gt_graph_dense_spacenet.p",'rb'))
        
        coord_transform = lambda v : np.stack([v[:, 1], 400 - v[:, 0]], axis=1)
        # coord_transform = lambda v : v[:, ::-1]
    rgb = read_rgb_img(rgb_path)
    config = addict.Dict()
    config.PATCH_SIZE = 256
    config.ROAD_NMS_RADIUS = 16
    config.TOPO_SAMPLE_NUM = 4
    config.NEIGHBOR_RADIUS = 64
    config.MAX_NEIGHBOR_QUERIES = 16
    gen = GraphLabelGenerator(config, gt_graph, coord_transform)
    patch = ((x0, y0), (x1, y1)) = ((64, 64), (64+config.PATCH_SIZE, 64+config.PATCH_SIZE))
    test_num = 64
    for i in range(test_num):
        rot_index = np.random.randint(0, 4)
        points, samples = gen.sample_patch(patch, rot_index=rot_index)
        rgb_patch = rgb[y0:y1, x0:x1, ::-1].copy()
        rgb_patch = np.rot90(rgb_patch, rot_index, (0, 1)).copy()
        for pairs, shall_connect, valid in samples:
            color = tuple(int(c) for c in np.random.randint(0, 256, size=3))

            for (src, tgt), connected, is_valid in zip(pairs, shall_connect, valid):
                if not is_valid:
                    continue
                p0, p1 = points[src], points[tgt]
                cv2.circle(rgb_patch, p0.astype(np.int32), 4, color, -1)
                cv2.circle(rgb_patch, p1.astype(np.int32), 2, color, -1)
                if connected:
                    cv2.line(
                        rgb_patch,
                        (int(p0[0]), int(p0[1])),
                        (int(p1[0]), int(p1[1])),
                        (255, 255, 255),
                        1,
                    )
        cv2.imwrite(f'debug/viz_{i}.png', rgb_patch)

        
def graph_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'graph_points':
            tensors = [item[key] for item in batch]
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            collated[key] = torch.stack(padded, dim=0)
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated


def deepglobe_data_partition(data_dir):
    # 获取所有图像ID
    all_ids = [os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(data_dir, '*_sat.jpg'))]
    all_ids = sorted(list(set(all_ids)))
    
    # 按8:1:1的比例划分训练集、验证集和测试集
    n = len(all_ids)
    train_ids = all_ids[:int(0.8*n)]
    val_ids = all_ids[int(0.8*n):int(0.9*n)]
    test_ids = all_ids[int(0.9*n):]
    
    return train_ids, val_ids, test_ids


class DeepGlobeDataset(Dataset):
    def __init__(self, config, is_train, dev_run=False):
        self.config = config
        self.is_train = is_train
        
        # 设置日志
        self.logger = logging.getLogger('DeepGlobeDataset')
        
        # 数据集路径
        self.data_dir = config.DATA_DIR
        self.IMAGE_SIZE = config.IMAGE_SIZE
        self.SAMPLE_MARGIN = config.SAMPLE_MARGIN
        self.PATCH_SIZE = config.PATCH_SIZE
        
        self.logger.info(f"初始化DeepGlobe数据集 - {'训练' if is_train else '验证'}模式")
        self.logger.info(f"数据集根目录: {self.data_dir}")
        
        # 获取数据集划分
        train_ids, val_ids, test_ids = deepglobe_data_partition(self.data_dir)
        self.tile_ids = train_ids if self.is_train else val_ids
        
        # 存储所有图像路径
        self.rgb_paths = []
        self.mask_paths = []
        
        ##### FAST DEBUG
        if dev_run:
            self.tile_ids = self.tile_ids[:4]
            self.logger.info("开发模式: 仅使用前4个样本")
        ##### FAST DEBUG
        
        valid_count = 0
        skipped_count = 0
        for tile_id in self.tile_ids:
            rgb_path = os.path.join(self.data_dir, f'{tile_id}_sat.jpg')
            mask_path = os.path.join(self.data_dir, f'{tile_id}_mask.png')
            
            if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
                self.logger.warning(f'跳过缺失的样本 {tile_id}')
                skipped_count += 1
                continue
                
            self.rgb_paths.append(rgb_path)
            self.mask_paths.append(mask_path)
            valid_count += 1
        
        if valid_count == 0:
            raise RuntimeError(f"在 {self.data_dir} 中没有找到有效的图像")
        
        self.logger.info(f"数据集统计:")
        self.logger.info(f"  - 总样本数: {len(self.tile_ids)}")
        self.logger.info(f"  - 有效样本: {valid_count}")
        self.logger.info(f"  - 跳过样本: {skipped_count}")
        
        self.sample_min = self.SAMPLE_MARGIN
        self.sample_max = self.IMAGE_SIZE - (self.PATCH_SIZE + self.SAMPLE_MARGIN)
        
        if not self.is_train:
            eval_patches_per_edge = math.ceil((self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.PATCH_SIZE)
            self.eval_patches = []
            for i in range(len(self.rgb_paths)):
                self.eval_patches += get_patch_info_one_img(
                    i, self.IMAGE_SIZE, self.SAMPLE_MARGIN, self.PATCH_SIZE, eval_patches_per_edge
                )
            self.logger.info(f"验证模式: 每个图像生成 {eval_patches_per_edge}x{eval_patches_per_edge} 个patch")
            self.logger.info(f"总验证patch数: {len(self.eval_patches)}")
    
    def __len__(self):
        if self.is_train:
            return 10000  # 训练时返回固定数量的样本
        else:
            return len(self.eval_patches)
    
    def __getitem__(self, idx):
        try:
            if self.is_train:
                img_idx = np.random.randint(low=0, high=len(self.rgb_paths))
                begin_x = np.random.randint(low=self.sample_min, high=self.sample_max+1)
                begin_y = np.random.randint(low=self.sample_min, high=self.sample_max+1)
                end_x, end_y = begin_x + self.PATCH_SIZE, begin_y + self.PATCH_SIZE
                
                # 读取图像和掩码
                rgb = read_rgb_img(self.rgb_paths[img_idx])
                mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)
                
                if rgb is None or mask is None:
                    self.logger.error(f"加载失败: {self.rgb_paths[img_idx]} 或 {self.mask_paths[img_idx]}")
                    # 返回一个有效的默认值
                    rgb = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE, 3), dtype=np.uint8)
                    mask = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=np.uint8)
                
                # 裁剪patch
                rgb_patch = rgb[begin_y:end_y, begin_x:end_x, :]
                mask_patch = mask[begin_y:end_y, begin_x:end_x]
                
                # 数据增强
                rot_index = np.random.randint(0, 4)
                rgb_patch = np.rot90(rgb_patch, rot_index, [0,1]).copy()
                mask_patch = np.rot90(mask_patch, rot_index, [0, 1]).copy()
                
                # 将掩码转换为二值掩码（道路和背景）
                road_mask = (mask_patch > 0).astype(np.float32)
                keypoint_mask = (mask_patch == 255).astype(np.float32)  # 假设255表示关键点
                
                # 调试信息
                if idx % 1000 == 0:  # 每1000个样本打印一次
                    self.logger.debug(f"训练样本 {idx}:")
                    self.logger.debug(f"  - 图像: {os.path.basename(self.rgb_paths[img_idx])}")
                    self.logger.debug(f"  - 位置: ({begin_x}, {begin_y}) -> ({end_x}, {end_y})")
                    self.logger.debug(f"  - 旋转: {rot_index * 90}度")
                    self.logger.debug(f"  - 道路像素比例: {road_mask.mean():.2%}")
                    self.logger.debug(f"  - 关键点像素比例: {keypoint_mask.mean():.2%}")
                
                return {
                    'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
                    'keypoint_mask': torch.tensor(keypoint_mask, dtype=torch.float32),
                    'road_mask': torch.tensor(road_mask, dtype=torch.float32),
                }
            else:
                # Returns eval patch
                img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]
                
                # 读取图像和掩码
                rgb = read_rgb_img(self.rgb_paths[img_idx])
                mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)
                
                if rgb is None or mask is None:
                    self.logger.error(f"加载失败: {self.rgb_paths[img_idx]} 或 {self.mask_paths[img_idx]}")
                    rgb = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE, 3), dtype=np.uint8)
                    mask = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=np.uint8)
                
                # 裁剪patch
                rgb_patch = rgb[begin_y:end_y, begin_x:end_x, :]
                mask_patch = mask[begin_y:end_y, begin_x:end_x]
                
                # 将掩码转换为二值掩码
                road_mask = (mask_patch > 0).astype(np.float32)
                keypoint_mask = (mask_patch == 255).astype(np.float32)
                
                # 调试信息
                if idx % 100 == 0:  # 每100个样本打印一次
                    self.logger.debug(f"验证样本 {idx}:")
                    self.logger.debug(f"  - 图像: {os.path.basename(self.rgb_paths[img_idx])}")
                    self.logger.debug(f"  - 位置: ({begin_x}, {begin_y}) -> ({end_x}, {end_y})")
                    self.logger.debug(f"  - 道路像素比例: {road_mask.mean():.2%}")
                    self.logger.debug(f"  - 关键点像素比例: {keypoint_mask.mean():.2%}")
                
                return {
                    'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
                    'keypoint_mask': torch.tensor(keypoint_mask, dtype=torch.float32),
                    'road_mask': torch.tensor(road_mask, dtype=torch.float32),
                }
        except Exception as e:
            self.logger.error(f"处理样本 {idx} 时发生错误: {str(e)}")
            # 返回一个有效的默认值
            return {
                'rgb': torch.zeros((self.PATCH_SIZE, self.PATCH_SIZE, 3), dtype=torch.float32),
                'keypoint_mask': torch.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=torch.float32),
                'road_mask': torch.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=torch.float32),
            }


class SatMapDataset(Dataset):
    def __init__(self, config, is_train, dev_run=False):
        self.config = config
        
        assert self.config.DATASET in {'cityscale', 'spacenet', 'deepglobe'}
        if self.config.DATASET == 'deepglobe':
            self.IMAGE_SIZE = 1024
            self.SAMPLE_MARGIN = 64
            
            # 获取数据集划分
            train_ids, val_ids, test_ids = deepglobe_data_partition(self.config.DATA_DIR)
            self.tile_ids = train_ids if is_train else val_ids
            
            # 存储所有图像路径
            self.rgb_paths = []
            self.mask_paths = []
            
            for tile_id in self.tile_ids:
                rgb_path = os.path.join(self.config.DATA_DIR, f'{tile_id}_sat.jpg')
                mask_path = os.path.join(self.config.DATA_DIR, f'{tile_id}_mask.png')
                
                if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
                    print(f'===== skipped missing tile {tile_id} =====')
                    continue
                    
                self.rgb_paths.append(rgb_path)
                self.mask_paths.append(mask_path)
            
            if len(self.rgb_paths) == 0:
                raise RuntimeError(f"No valid images found in {self.config.DATA_DIR}")
                
        elif self.config.DATASET == 'cityscale':
            self.IMAGE_SIZE = 2048
            self.SAMPLE_MARGIN = 64

            rgb_pattern = './cityscale/20cities/region_{}_sat.png'
            keypoint_mask_pattern = './cityscale/processed/keypoint_mask_{}.png'
            road_mask_pattern = './cityscale/processed/road_mask_{}.png'
            gt_graph_pattern = './cityscale/20cities/region_{}_refine_gt_graph.p'
            
            train, val, test = cityscale_data_partition()

            # coord-transform = (r, c) -> (x, y)
            # takes [N, 2] points
            coord_transform = lambda v : v[:, ::-1]

        elif self.config.DATASET == 'spacenet':
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = 0

            rgb_pattern = './spacenet/RGB_1.0_meter/{}__rgb.png'
            keypoint_mask_pattern = './spacenet/processed/keypoint_mask_{}.png'
            road_mask_pattern = './spacenet/processed/road_mask_{}.png'
            gt_graph_pattern = './spacenet/RGB_1.0_meter/{}__gt_graph.p'
            
            train, val, test = spacenet_data_partition()

            # coord-transform ??? -> (x, y)
            # takes [N, 2] points
            coord_transform = lambda v : np.stack([v[:, 1], 400 - v[:, 0]], axis=1)

        self.is_train = is_train

        if self.config.DATASET != 'deepglobe':
            train_split = train + val
            test_split = test

            tile_indices = train_split if self.is_train else test_split
            self.tile_indices = tile_indices
            
            # Stores all imgs in memory.
            self.rgbs, self.keypoint_masks, self.road_masks = [], [], []
            # For graph label generation.
            self.graph_label_generators = []
            
            ##### FAST DEBUG
            if dev_run:
                tile_indices = tile_indices[:4]
            ##### FAST DEBUG
            
            for tile_idx in tile_indices:
                print(f'loading tile {tile_idx}')
                rgb_path = rgb_pattern.format(tile_idx)
                road_mask_path = road_mask_pattern.format(tile_idx)
                keypoint_mask_path = keypoint_mask_pattern.format(tile_idx)

                # graph label gen
                # gt graph: dict for adj list, for cityscale set keys are (r, c) nodes, values are list of (r, c) nodes
                # I don't know what coord system spacenet uses but we convert them all to (x, y)
                gt_graph_adj = pickle.load(open(gt_graph_pattern.format(tile_idx),'rb'))
                if len(gt_graph_adj) == 0:
                    print(f'===== skipped empty tile {tile_idx} =====')
                    continue
                    
                self.rgbs.append(read_rgb_img(rgb_path))
                self.road_masks.append(cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE))
                self.keypoint_masks.append(cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE))
                graph_label_generator = GraphLabelGenerator(config, gt_graph_adj, coord_transform)
                self.graph_label_generators.append(graph_label_generator)
        
        self.sample_min = self.SAMPLE_MARGIN
        self.sample_max = self.IMAGE_SIZE - (self.config.PATCH_SIZE + self.SAMPLE_MARGIN)

        if not self.is_train:
            eval_patches_per_edge = math.ceil((self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.config.PATCH_SIZE)
            self.eval_patches = []
            if self.config.DATASET == 'deepglobe':
                for i in range(len(self.rgb_paths)):
                    self.eval_patches += get_patch_info_one_img(
                        i, self.IMAGE_SIZE, self.SAMPLE_MARGIN, self.config.PATCH_SIZE, eval_patches_per_edge
                    )
            else:
                for i in range(len(tile_indices)):
                    self.eval_patches += get_patch_info_one_img(
                        i, self.IMAGE_SIZE, self.SAMPLE_MARGIN, self.config.PATCH_SIZE, eval_patches_per_edge
                    )

    def __len__(self):
        if self.is_train:
            if self.config.DATASET == 'deepglobe':
                return 10000  # 训练时返回固定数量的样本
            elif self.config.DATASET == 'cityscale':
                return max(1, int(self.IMAGE_SIZE / self.config.PATCH_SIZE)) ** 2 * 2500
            elif self.config.DATASET == 'spacenet':
                return 84667
        else:
            return len(self.eval_patches)

    def __getitem__(self, idx):
        try:
            # 获取图像和掩码路径
            rgb_path = self.rgb_paths[idx]
            keypoint_mask_path = self.keypoint_mask_paths[idx]
            road_mask_path = self.road_mask_paths[idx]
            
            # 读取图像和掩码
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                raise ValueError(f"无法读取图像: {rgb_path}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            keypoint_mask = cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE)
            if keypoint_mask is None:
                raise ValueError(f"无法读取关键点掩码: {keypoint_mask_path}")
            
            road_mask = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)
            if road_mask is None:
                raise ValueError(f"无法读取道路掩码: {road_mask_path}")
            
            # 确保所有图像尺寸一致
            if rgb.shape[:2] != keypoint_mask.shape[:2] or rgb.shape[:2] != road_mask.shape[:2]:
                raise ValueError(f"图像尺寸不匹配: RGB={rgb.shape}, Keypoint={keypoint_mask.shape}, Road={road_mask.shape}")
            
            # 转换为张量
            rgb = torch.from_numpy(rgb).float() / 255.0
            keypoint_mask = torch.from_numpy(keypoint_mask).float() / 255.0
            road_mask = torch.from_numpy(road_mask).float() / 255.0
            
            # 训练模式下的数据增强
            if self.is_train:
                # 随机裁剪
                h, w = rgb.shape[:2]
                if h > self.patch_size and w > self.patch_size:
                    top = torch.randint(0, h - self.patch_size, (1,)).item()
                    left = torch.randint(0, w - self.patch_size, (1,)).item()
                    rgb = rgb[top:top + self.patch_size, left:left + self.patch_size]
                    keypoint_mask = keypoint_mask[top:top + self.patch_size, left:left + self.patch_size]
                    road_mask = road_mask[top:top + self.patch_size, left:left + self.patch_size]
                else:
                    # 如果图像太小，进行填充
                    rgb = F.interpolate(rgb.permute(2, 0, 1).unsqueeze(0), 
                                      size=(self.patch_size, self.patch_size), 
                                      mode='bilinear', 
                                      align_corners=False).squeeze(0).permute(1, 2, 0)
                    keypoint_mask = F.interpolate(keypoint_mask.unsqueeze(0).unsqueeze(0), 
                                               size=(self.patch_size, self.patch_size), 
                                               mode='nearest').squeeze(0).squeeze(0)
                    road_mask = F.interpolate(road_mask.unsqueeze(0).unsqueeze(0), 
                                           size=(self.patch_size, self.patch_size), 
                                           mode='nearest').squeeze(0).squeeze(0)
                
                # 随机水平翻转
                if torch.rand(1) < 0.5:
                    rgb = torch.flip(rgb, [1])
                    keypoint_mask = torch.flip(keypoint_mask, [1])
                    road_mask = torch.flip(road_mask, [1])
                
                # 随机垂直翻转
                if torch.rand(1) < 0.5:
                    rgb = torch.flip(rgb, [0])
                    keypoint_mask = torch.flip(keypoint_mask, [0])
                    road_mask = torch.flip(road_mask, [0])
            else:
                # 验证模式下，确保图像尺寸正确
                if rgb.shape[0] != self.patch_size or rgb.shape[1] != self.patch_size:
                    rgb = F.interpolate(rgb.permute(2, 0, 1).unsqueeze(0), 
                                      size=(self.patch_size, self.patch_size), 
                                      mode='bilinear', 
                                      align_corners=False).squeeze(0).permute(1, 2, 0)
                    keypoint_mask = F.interpolate(keypoint_mask.unsqueeze(0).unsqueeze(0), 
                                               size=(self.patch_size, self.patch_size), 
                                               mode='nearest').squeeze(0).squeeze(0)
                    road_mask = F.interpolate(road_mask.unsqueeze(0).unsqueeze(0), 
                                           size=(self.patch_size, self.patch_size), 
                                           mode='nearest').squeeze(0).squeeze(0)
            
            return {
                'rgb': rgb,
                'keypoint_mask': keypoint_mask,
                'road_mask': road_mask
            }
        
        except Exception as e:
            logging.error(f"处理样本 {idx} 时发生错误: {str(e)}")
            # 返回一个有效的默认值
            default_rgb = torch.zeros((self.patch_size, self.patch_size, 3))
            default_mask = torch.zeros((self.patch_size, self.patch_size))
            return {
                'rgb': default_rgb,
                'keypoint_mask': default_mask,
                'road_mask': default_mask
            }


if __name__ == '__main__':
    test_graph_label_generator()
    # train, val, test = cityscale_data_partition()
    # print(f'cityscale train {len(train)} val {len(val)} test {len(test)}')
    # train, val, test = spacenet_data_partition()
    # print(f'spacenet train {len(train)} val {len(val)} test {len(test)}')
