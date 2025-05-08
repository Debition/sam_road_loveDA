import torch
import torch.nn.functional as F
from torch import nn

# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy

from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve

import lightning.pytorch as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.mask_decoder import MaskDecoder
from sam.segment_anything.modeling.prompt_encoder import PromptEncoder
from sam.segment_anything.modeling.transformer import TwoWayTransformer
from sam.segment_anything.modeling.common import LayerNorm2d

import wandb
import pprint
import torchvision

# Only needed for the ablation experiment of using a ViT-B model without SA-1B pre-training.
# It depends on detectron2 library. Not super important. 
# import vitdet


class SAMRoad(pl.LightningModule):
    """This is the RelationFormer module that performs object detection"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        assert config.MODEL_VERSION in {'vit_b', 'vit_l', 'vit_h'}
        if config.MODEL_VERSION == 'vit_b':
            ### SAM config (B)
            encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]
            ###
        elif config.MODEL_VERSION == 'vit_l':
            ### SAM config (L)
            encoder_embed_dim=1024
            encoder_depth=24
            encoder_num_heads=16
            encoder_global_attn_indexes=[5, 11, 17, 23]
            ###
        elif config.MODEL_VERSION == 'vit_h':
            ### SAM config (H)
            encoder_embed_dim=1280
            encoder_depth=32
            encoder_num_heads=16
            encoder_global_attn_indexes=[7, 15, 23, 31]
            ###
            
        prompt_embed_dim = 256
        # SAM default is 1024
        image_size = config.IMAGE_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        encoder_output_dim = prompt_embed_dim

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        if hasattr(config, 'NO_SAM') and config.NO_SAM:
            raise NotImplementedError((
                "This ablation experiment depends on detectron2, "
                "which is a bit messy and is not super important, "
                "so not including in the release. "
                "If you are interested, feel free to uncomment."))
        else:
            ### SAM vitb
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
                out_chans=prompt_embed_dim
            )

        if config.USE_SAM_DECODER:
            # SAM DECODER
            # Not used, just produce null embeddings
            self.prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            self.mask_decoder=MaskDecoder(
                num_multimask_outputs=2, # keypoint, road
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        else:
            #### Naive decoder
            activation = nn.GELU
            self.map_decoder = nn.Sequential(
                nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
            )

        #### Losses
        if hasattr(config, 'FOCAL_LOSS') and config.FOCAL_LOSS:
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()

        #### Metrics
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        # testing only, not used in training
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)

        if hasattr(config, 'NO_SAM') and config.NO_SAM:
            return
        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f)

            ## Resize pos embeddings, if needed
            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
                ckpt_state_dict = new_state_dict
            
            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params ######")
            pprint.pprint(mismatch_names)

            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)

    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        new_state_dict = {k : v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            # Copied from SAMed
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict

    def forward(self, rgb):
        # rgb: [B, H, W, C]
        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        
        # 确保输入图像尺寸正确
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # 获取图像嵌入
        image_embeddings = self.image_encoder(x)
        
        # 解码得到道路和关键点预测
        logits = self.map_decoder(image_embeddings)
        
        # 调整回原始尺寸
        if logits.shape[-2:] != (rgb.shape[1], rgb.shape[2]):
            logits = F.interpolate(logits, size=(rgb.shape[1], rgb.shape[2]), mode='bilinear', align_corners=False)
        
        # [B, H, W, 2]
        logits = logits.permute(0, 2, 3, 1)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        logits = self(rgb)
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(logits, gt_masks)
        
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=False, prog_bar=True)
        return mask_loss

    def validation_step(self, batch, batch_idx):
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        logits = self(rgb)
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(logits, gt_masks)
        
        self.log('val_mask_loss', mask_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 记录图像
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :]
            viz_pred_keypoint = torch.sigmoid(logits[:max_viz_num, :, :, 0])
            viz_pred_road = torch.sigmoid(logits[:max_viz_num, :, :, 1])
            viz_gt_keypoint = keypoint_mask[:max_viz_num, ...]
            viz_gt_road = road_mask[:max_viz_num, ...]
            
            columns = ['rgb', 'gt_keypoint', 'gt_road', 'pred_keypoint', 'pred_road']
            data = [[wandb.Image(x.cpu().numpy()) for x in row] for row in list(zip(viz_rgb, viz_gt_keypoint, viz_gt_road, viz_pred_keypoint, viz_pred_road))]
            self.logger.log_table(key='viz_table', columns=columns, data=data)

        self.keypoint_iou.update(torch.sigmoid(logits[..., 0]), keypoint_mask)
        self.road_iou.update(torch.sigmoid(logits[..., 1]), road_mask)

    def on_validation_epoch_end(self):
        keypoint_iou = self.keypoint_iou.compute()
        road_iou = self.road_iou.compute()
        self.log("keypoint_iou", keypoint_iou)
        self.log("road_iou", road_iou)
        self.keypoint_iou.reset()
        self.road_iou.reset()

    def test_step(self, batch, batch_idx):
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        logits = self(rgb)

        self.keypoint_pr_curve.update(torch.sigmoid(logits[..., 0]), keypoint_mask.to(torch.int32))
        self.road_pr_curve.update(torch.sigmoid(logits[..., 1]), road_mask.to(torch.int32))

    def on_test_end(self):
        def find_best_threshold(pr_curve_metric, category):
            print(f'======= {category} ======')   
            precision, recall, thresholds = pr_curve_metric.compute()
            f1_scores = 2 * (precision * recall) / (precision + recall)
            best_threshold_index = torch.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_index]
            best_precision = precision[best_threshold_index]
            best_recall = recall[best_threshold_index]
            best_f1 = f1_scores[best_threshold_index]
            print(f'Best threshold {best_threshold}, P={best_precision} R={best_recall} F1={best_f1}')
        
        print('======= Finding best thresholds ======')
        find_best_threshold(self.keypoint_pr_curve, 'keypoint')
        find_best_threshold(self.road_pr_curve, 'road')

    def configure_optimizers(self):
        param_dicts = []

        if not hasattr(self.config, 'FREEZE_ENCODER') or not self.config.FREEZE_ENCODER:
            if not hasattr(self.config, 'USE_LORA') or not self.config.USE_LORA:
                encoder_params = {
                    'params': [p for k, p in self.image_encoder.named_parameters() if 'image_encoder.'+k in self.matched_param_names],
                    'lr': self.config.BASE_LR * (self.config.ENCODER_LR_FACTOR if hasattr(self.config, 'ENCODER_LR_FACTOR') else 1.0),
                }
                param_dicts.append(encoder_params)
        if hasattr(self.config, 'USE_LORA') and self.config.USE_LORA:
            # LoRA params only
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'qkv.linear_' in k],
                'lr': self.config.BASE_LR,
            }
            param_dicts.append(encoder_params)
        
        if self.config.USE_SAM_DECODER:
            matched_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if 'mask_decoder.'+k in self.matched_param_names],
                'lr': self.config.BASE_LR * 0.1
            }
            fresh_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if 'mask_decoder.'+k not in self.matched_param_names],
                'lr': self.config.BASE_LR
            }
            decoder_params = [matched_decoder_params, fresh_decoder_params]
        else:
            decoder_params = [{
                'params': [p for p in self.map_decoder.parameters()],
                'lr': self.config.BASE_LR
            }]
        param_dicts += decoder_params
        
        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict['params']])
            print(f'optim param dict {i} params num: {param_num}')

        optimizer = torch.optim.Adam(param_dicts, lr=self.config.BASE_LR)
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9,], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': step_lr}

