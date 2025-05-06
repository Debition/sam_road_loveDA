import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
import pprint
from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, JaccardIndex, F1Score, PrecisionRecallCurve

import lightning.pytorch as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.mask_decoder import MaskDecoder
from sam.segment_anything.modeling.prompt_encoder import PromptEncoder
from sam.segment_anything.modeling.transformer import TwoWayTransformer
from sam.segment_anything.modeling.common import LayerNorm2d

import wandb
import numpy as np
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SAMAISD')

class _LoRA_qkv(nn.Module):
    """在SAM中的实现:
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

class SAMAISD(pl.LightningModule):
    """基于SAM的遥感建筑物和道路分割模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config'])
        
        # 添加辅助函数来处理配置访问
        def get_config(key, default=None):
            return getattr(config, key, default)
        self.get_config = get_config
        
        logger.info(f"初始化SAM模型，配置: {config}")

        assert config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}, f"不支持的SAM版本: {config.SAM_VERSION}"
        if config.SAM_VERSION == 'vit_b':
            encoder_embed_dim = 768
            encoder_depth = 12
            encoder_num_heads = 12
            encoder_global_attn_indexes = [2, 5, 8, 11]
        elif config.SAM_VERSION == 'vit_l':
            encoder_embed_dim = 1024
            encoder_depth = 24
            encoder_num_heads = 16
            encoder_global_attn_indexes = [5, 11, 17, 23]
        else:  # vit_h
            encoder_embed_dim = 1280
            encoder_depth = 32
            encoder_num_heads = 16
            encoder_global_attn_indexes = [7, 15, 23, 31]
        logger.info(f"使用{config.SAM_VERSION}版本, embed_dim={encoder_embed_dim}, depth={encoder_depth}")
            
        prompt_embed_dim = 256
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        logger.info(f"图像尺寸: {image_size}x{image_size}, 特征图尺寸: {image_embedding_size}x{image_embedding_size}")

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        # SAM图像编码器
        self.image_encoder = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )

        # 提示编码器
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        # 默认冻结prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        self.num_classes = config.NUM_CLASSES
        
        # 使用自定义解码器
        self.decoder = EnhancedDecoder(
            encoder_dim=prompt_embed_dim,
            num_classes=self.num_classes,
            config=config
        )

        # 应用LoRA
        if config.ENCODER_LORA:
            self._apply_lora()

        # 初始化指标
        self.mean_iou = JaccardIndex(task="multiclass", num_classes=config.NUM_CLASSES, ignore_index=0)
        self.class_ious = nn.ModuleList([
            BinaryJaccardIndex(threshold=0.5) for _ in range(1, config.NUM_CLASSES)  # 跳过背景类(0)
        ])
        self.class_f1s = nn.ModuleList([
            F1Score(task="binary", threshold=0.5) for _ in range(1, config.NUM_CLASSES)  # 跳过背景类(0)
        ])
        
        # 测试专用指标
        self.class_pr_curves = nn.ModuleList([
            PrecisionRecallCurve(task="binary", ignore_index=-1) for _ in range(1, config.NUM_CLASSES)  # 跳过背景类(0)
        ])

        # 用于存储验证指标
        self.val_mean_iou_list = []
        self.val_class_ious_list = [[] for _ in range(1, config.NUM_CLASSES)]

        # 加载预训练权重
        if not self.get_config('NO_PRETRAIN', False):
            checkpoint_path = self.get_config('SAM_CKPT_PATH')
            if checkpoint_path:
                print(f"Loading SAM weights from {checkpoint_path}")
                with open(checkpoint_path, "rb") as f:
                    state_dict = torch.load(f)
                    
                # 调整位置编码
                if image_size != 1024:
                    state_dict = self._resize_pos_embed(state_dict, 
                                                     image_size, 
                                                     vit_patch_size, 
                                                     encoder_global_attn_indexes)
                
                # 记录匹配和不匹配的参数
                matched_names = []
                mismatch_names = []
                state_dict_to_load = {}
                
                for k, v in self.named_parameters():
                    if k in state_dict and v.shape == state_dict[k].shape:
                        matched_names.append(k)
                        state_dict_to_load[k] = state_dict[k]
                    else:
                        mismatch_names.append(k)
                
                print("###### Matched params ######")
                pprint.pprint(matched_names)
                print("###### Mismatched params ######")
                pprint.pprint(mismatch_names)
                
                self.matched_param_names = set(matched_names)
                self.load_state_dict(state_dict_to_load, strict=False)
            else:
                print("No checkpoint path provided, using random initialization")

        # 打印参数统计信息
        self._print_param_stats()

    def _apply_lora(self):
        """应用LoRA到编码器"""
        r = self.get_config('LORA_RANK', 4)
        lora_layer_selection = None
        assert r > 0
        if lora_layer_selection:
            self.lora_layer_selection = lora_layer_selection
        else:
            self.lora_layer_selection = list(
                range(len(self.image_encoder.blocks)))  # 默认对所有图像编码器层应用LoRA
        # 创建存储空间，然后初始化或加载权重
        self.w_As = []  # 线性层
        self.w_Bs = []

        # 先冻结参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # 进行LoRA手术
        for t_layer_i, blk in enumerate(self.image_encoder.blocks):
            # 如果我们只想对部分层应用LoRA
            if t_layer_i not in self.lora_layer_selection:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        # 初始化LoRA参数
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

            
        # 应用LoRA到每个Transformer块
        for block in self.image_encoder.blocks:
            # 获取原始QKV层的维度
            dim = block.attn.qkv.in_features
            num_heads = block.attn.num_heads
            block.attn.qkv.num_heads = num_heads  # 保存num_heads供LoRA使用
            
            # 创建LoRA层
            w_a_q = nn.Linear(dim, self.get_config('LORA_RANK', 4), bias=False)
            w_b_q = nn.Linear(self.get_config('LORA_RANK', 4), dim, bias=False)
            w_a_v = nn.Linear(dim, self.get_config('LORA_RANK', 4), bias=False)
            w_b_v = nn.Linear(self.get_config('LORA_RANK', 4), dim, bias=False)
            
            # 初始化
            nn.init.kaiming_uniform_(w_a_q.weight, a=math.sqrt(5))
            nn.init.zeros_(w_b_q.weight)
            nn.init.kaiming_uniform_(w_a_v.weight, a=math.sqrt(5))
            nn.init.zeros_(w_b_v.weight)
            
            # 替换原始QKV层
            block.attn.qkv = _LoRA_qkv(
                block.attn.qkv,
                w_a_q,
                w_b_q,
                w_a_v,
                w_b_v,
            )
        
        # 统计LoRA参数
        lora_params = sum(p.numel() for name, p in self.named_parameters() if 'linear_a' in name or 'linear_b' in name)
        logger.info(f"LoRA参数总数: {lora_params:,}")

    def _resize_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        """调整位置编码大小"""
        pos_embed = state_dict['image_encoder.pos_embed']
        old_size = int(math.sqrt(pos_embed.shape[1]))
        new_size = image_size // vit_patch_size
        
        if old_size != new_size:
            # 调整位置编码
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)
            state_dict['image_encoder.pos_embed'] = pos_embed
            
            # 调整相对位置编码
            for idx in encoder_global_attn_indexes:
                for key in [f'image_encoder.blocks.{idx}.attn.rel_pos_h', f'image_encoder.blocks.{idx}.attn.rel_pos_w']:
                    rel_pos = state_dict[key]
                    new_rel_pos = F.interpolate(
                        rel_pos.unsqueeze(0).unsqueeze(0),
                        size=(2 * new_size - 1),
                        mode='linear',
                        align_corners=False
                    )
                    state_dict[key] = new_rel_pos[0, 0]
        
        return state_dict

    def forward(self, batch):
        # 获取输入图像
        image = batch['image']
        
        # 确保输入维度正确
        if len(image.shape) != 4:  # [B, C, H, W]
            raise ValueError(f"Expected 4D input tensor (B,C,H,W), got shape {image.shape}")
        
        # 归一化图像
        x = (image - self.pixel_mean) / self.pixel_std
        
        # 获取图像特征
        features = self.image_encoder(x)
        
        # 使用解码器
        outputs = self.decoder(features)
        
        return outputs

    def training_step(self, batch, batch_idx):
        # 前向传播
        outputs = self.forward(batch)
        masks = batch['mask']
        
        # 计算损失
        loss = compute_loss(outputs['main'], masks, self.config)
        
        # 计算IoU
        preds = torch.argmax(outputs['main'], dim=1)
        iou = self.mean_iou(preds, masks)
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        
        # 每N步记录预测可视化
        if batch_idx % self.get_config('LOG_INTERVAL', 100) == 0:
            self._log_images(batch, outputs['main'])
        
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.fast_dev_run:
            return
            
        # 前向传播
        outputs = self.forward(batch)
        mask_logits = outputs['main']
        
        # 获取目标掩码
        masks = batch['mask']
        
        # 计算损失
        loss = compute_loss(mask_logits, masks, self.config)
        
        # 计算预测结果
        preds = torch.argmax(mask_logits, dim=1)
        
        # 更新验证指标
        mean_iou = self.mean_iou(preds, masks)
        
        # 计算每个类别的IoU
        class_scores = {}
        for cls_idx in range(1, self.num_classes):
            self.class_ious[cls_idx-1].update(preds == cls_idx, masks == cls_idx)
            class_scores[f'val_iou_class_{cls_idx}'] = self.class_ious[cls_idx-1].compute()
        
        # 记录指标
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mean_iou', mean_iou, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in class_scores.items():
            self.log(k, v, on_step=False, on_epoch=True)
        
        # 如果是第一个批次，记录一些预览图像
        if batch_idx == 0:
            self._log_images(batch, mask_logits)

        return {'val_loss': loss, 'val_mean_iou': mean_iou, 'class_scores': class_scores}

    def configure_optimizers(self):
        # 基于不同的组件使用不同的学习率
        param_dicts = []
        
        # 图像编码器参数
        if hasattr(self.config, 'FREEZE_ENCODER') and self.config.FREEZE_ENCODER:
            # 编码器已在前面冻结
            pass
        elif hasattr(self.config, 'ENCODER_LORA') and self.config.ENCODER_LORA:
            # 只优化LoRA参数
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'qkv.linear_' in k],
                'lr': self.config.BASE_LR,
            }
            param_dicts.append(encoder_params)
        else:
            # 全部微调编码器
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if hasattr(self, 'matched_param_names') and 'image_encoder.'+k in self.matched_param_names],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
            }
            param_dicts.append(encoder_params)
        
        # 解码器参数
        decoder_params = [{
            'params': [p for p in self.decoder.parameters()],
            'lr': self.config.BASE_LR
        }]
        param_dicts += decoder_params
        
        # 打印参数数量信息
        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict['params']])
            print(f'optim param dict {i} params num: {param_num}')
        
        # 添加梯度裁剪
        for param_dict in param_dicts:
            param_dict['max_norm'] = self.config.GRAD_CLIP
        
        # 使用AdamW优化器，添加权重衰减
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.config.BASE_LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 获取dataloader长度，如果在fast_dev_run模式下使用默认值
        try:
            steps_per_epoch = len(self.trainer.train_dataloader)
        except:
            steps_per_epoch = 100  # fast_dev_run的默认值
        
        # 使用带预热的学习率调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.BASE_LR,
            epochs=self.config.TRAIN_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"  # 每个步骤都更新学习率
            }
        }

    def _log_images(self, batch, logits, max_images=4):
        """记录图像用于可视化"""
        images = batch['image'][:max_images]
        masks = batch['mask'][:max_images]
        preds = torch.argmax(logits[:max_images], dim=1)
        
        # 修正颜色映射，确保与类别对应
        color_map = [
            [255, 255, 255], # 0: 背景 (白色)
            [255, 0, 0],     # 1: 建筑 (红色)
            [0, 0, 255],     # 2: 道路 (蓝色)
        ]
        
        # 为Wandb创建可视化表格
        data = []
        columns = ["原图", "真实掩码", "预测掩码"]
        
        for i in range(len(images)):
            # 获取图像和掩码
            img = images[i].permute(1, 2, 0).cpu().numpy()
            # 反归一化图像
            img = img * self.pixel_std.cpu().numpy() + self.pixel_mean.cpu().numpy()
            img = img.clip(0, 255).astype(np.uint8)
            
            # 获取真实和预测掩码
            true_mask = masks[i].cpu().numpy()
            pred_mask = preds[i].cpu().numpy()
            
            # 创建彩色掩码
            true_color = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
            pred_color = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
            
            for c, color in enumerate(color_map):
                true_color[true_mask == c] = color
                pred_color[pred_mask == c] = color
            
            # 转换为Wandb图像
            img_wandb = wandb.Image(img)
            true_wandb = wandb.Image(true_color)
            pred_wandb = wandb.Image(pred_color)
            
            data.append([img_wandb, true_wandb, pred_wandb])
        
        # 记录图像表格
        self.logger.experiment.log({"预览图像": wandb.Table(columns=columns, data=data)})

    def _print_param_stats(self):
        """打印模型参数统计信息"""
        print("\n" + "="*60)
        print("模型参数统计")
        print("="*60)
        
        # 总参数
        total_params = sum(p.numel() for p in self.parameters())
        print(f"总参数数量: {total_params:,}")
        
        # 可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"可训练参数数量: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"冻结参数数量: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        
        # 按组件划分的参数统计
        encoder_params = sum(p.numel() for name, p in self.named_parameters() if 'image_encoder' in name)
        encoder_train_params = sum(p.numel() for name, p in self.named_parameters() if 'image_encoder' in name and p.requires_grad)
        print(f"\n图像编码器参数: {encoder_params:,}")
        print(f"  - 可训练: {encoder_train_params:,} ({encoder_train_params/encoder_params*100:.2f}%)")
        print(f"  - 冻结: {encoder_params-encoder_train_params:,} ({(encoder_params-encoder_train_params)/encoder_params*100:.2f}%)")
        
        # 解码器参数
        decoder_params = sum(p.numel() for name, p in self.named_parameters() if 'decoder' in name)
        decoder_train_params = sum(p.numel() for name, p in self.named_parameters() if 'decoder' in name and p.requires_grad)
        print(f"\n解码器参数: {decoder_params:,}")
        print(f"  - 可训练: {decoder_train_params:,} ({decoder_train_params/decoder_params*100:.2f}%)")
        print(f"  - 冻结: {decoder_params-decoder_train_params:,} ({(decoder_params-decoder_train_params)/decoder_params*100:.2f}%)")
        
        print("="*60)

# 从sam_loveda.py导入其他必要的类和函数
from sam_loveda import EnhancedDecoder, compute_loss

def focal_loss(pred, target, alpha=0.25, gamma=2.0, weights=None):
    """Focal Loss实现"""
    ce_loss = F.cross_entropy(pred, target, reduction='none', weight=weights)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean() 