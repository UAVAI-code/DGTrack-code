"""
Basic DGTrack model. DGTrack。
"""
import math
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.DGTrack.deit import deit_tiny_patch16_224, deit_tiny_patch16_224_distill
from lib.models.DGTrack.vision_transformer import vit_tiny_patch16_224
from lib.models.DGTrack.eva import eva02_tiny_patch14_224
from lib.utils.box_ops import box_xyxy_to_cxcywh

import timm
from lib.utils.box_ops import box_xywh_to_xyxy
from lib.models.DGTrack.loss_functions import DJSLoss
from lib.models.DGTrack.statistics_network import (
    GlobalStatisticsNetwork,
)

from torch.nn.functional import l1_loss


class DGTrack(nn.Module):
    """ This is the base class for DGTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
            self.feat_sz_t = int(box_head.feat_sz_t)
            self.feat_len_t = int(box_head.feat_sz_t ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.djs_loss = DJSLoss()

        self.feature_map_size = 8
        self.feature_map_channels = transformer.embed_dim
        self.num_ch_coding = self.backbone.embed_dim
        self.coding_size = 8

        self.global_stat_x = GlobalStatisticsNetwork(
            feature_map_size=self.feature_map_size,
            feature_map_channels=self.feature_map_channels,
            coding_channels=self.num_ch_coding,
            coding_size=self.coding_size,
        )

        self.l1_loss = l1_loss

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                template_anno: torch.Tensor,
                search_anno: torch.Tensor,
                return_last_attn=False,
                ):

        if self.training:
            template_anno = torch.round(template_anno * 8).int()
            template_anno[template_anno < 0] = 0
            search_anno = torch.round(search_anno * 16).int()
            search_anno[search_anno < 0] = 0

        x, aux_dict = self.backbone(z=template, x=search,
                                    return_last_attn=return_last_attn, )

        if self.training:
            prob_active_m = torch.cat(aux_dict['probs_active'], dim=1).mean(dim=1)
            prob_active_m = prob_active_m.reshape(len(prob_active_m), 1)
            expected_active_ratio = 0.3 * torch.ones(prob_active_m.shape)
            activeness_loss = self.l1_loss(prob_active_m, expected_active_ratio.to(prob_active_m.device))
        else:
            activeness_loss = torch.zeros(0)









        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None, template_anno=template_anno, search_anno=search_anno)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['activeness_loss'] = activeness_loss
        return out

    def forward_head(self, cat_feature, gt_score_map=None, template_anno=None, search_anno=None):
        """
           DGTrack 类中的 forward_head 方法，负责执行模型的前向传播头部（即分类器或回归器）的逻辑；
           接收参数：
           cat_feature: transformer 网络的输出特征。
           gt_score_map: 真实得分图（ground truth score map）。
           template_anno 和 search_anno: 分别是模板和搜索区域的标注
           cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
           """
        if self.training:
            feat_len_t = cat_feature.shape[1] - self.feat_len_s
            feat_sz_t = int(math.sqrt(feat_len_t))
            enc_opt_z = cat_feature[:, 0:feat_len_t]
            opt = (enc_opt_z.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat_z = opt.view(-1, C, feat_sz_t, feat_sz_t)

        enc_opt = cat_feature[:, -self.feat_len_s:]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        global_mutual_loss = torch.zeros(0)
        if self.training:
            batch_size = opt_feat.shape[0] // 3
            source_feat = opt_feat[:batch_size]
            augmented_feat1 = opt_feat[batch_size:2 * batch_size]
            augmented_feat2 = opt_feat[2 * batch_size:]

            opt_feat_mask = torch.zeros(batch_size, source_feat.shape[1], 8, 8)

            for i in range(batch_size):
                source_bbox = search_anno.squeeze()[i]
                source_bbox = torch.tensor(
                    [source_bbox[0], source_bbox[1], min([source_bbox[2], 8]), min([source_bbox[3], 8])])
                x_s = source_bbox[0]
                y_s = source_bbox[1]

                augmented_bbox1 = search_anno.squeeze()[i + batch_size]
                augmented_bbox1 = torch.tensor(
                    [augmented_bbox1[0], augmented_bbox1[1], min([augmented_bbox1[2], 8]),
                     min([augmented_bbox1[3], 8])])
                x_a1 = augmented_bbox1[0]
                y_a1 = augmented_bbox1[1]

                augmented_bbox2 = search_anno.squeeze()[i + 2 * batch_size]
                augmented_bbox2 = torch.tensor(
                    [augmented_bbox2[0], augmented_bbox2[1], min([augmented_bbox2[2], 8]),
                     min([augmented_bbox2[3], 8])])
                x_a2 = augmented_bbox2[0]
                y_a2 = augmented_bbox2[1]

                h = min(source_bbox[3], augmented_bbox1[3], augmented_bbox2[3])
                w = min(source_bbox[2], augmented_bbox1[2], augmented_bbox2[2])

                opt_feat_mask[i, :, y_s:y_s + h, x_s:x_s + w] = 1
                opt_feat_mask[i, :, y_a1:y_a1 + h, x_a1:x_a1 + w] = 1
                opt_feat_mask[i, :, y_a2:y_a2 + h, x_a2:x_a2 + w] = 1

            source_feat_masked = source_feat[:, :, :8, :8] * opt_feat_mask.to(source_feat.device)
            augmented_feat1_masked = augmented_feat1[:, :, :8, :8] * opt_feat_mask.to(augmented_feat1.device)
            augmented_feat2_masked = augmented_feat2[:, :, :8, :8] * opt_feat_mask.to(augmented_feat2.device)

            source_shuffled1 = torch.cat([source_feat_masked[1:], source_feat_masked[0].unsqueeze(0)], dim=0)
            source_shuffled2 = torch.cat([source_feat_masked[1:], source_feat_masked[0].unsqueeze(0)], dim=0)

            global_mutual_M_R_x1 = self.global_stat_x(source_feat_masked, augmented_feat1_masked)
            global_mutual_M_R_x_prime1 = self.global_stat_x(source_shuffled1, augmented_feat1_masked)

            global_mutual_M_R_x2 = self.global_stat_x(source_feat_masked, augmented_feat2_masked)
            global_mutual_M_R_x_prime2 = self.global_stat_x(source_shuffled2, augmented_feat2_masked)

            global_mutual_loss1 = self.djs_loss(
                T=global_mutual_M_R_x1,
                T_prime=global_mutual_M_R_x_prime1,
            )

            global_mutual_loss2 = self.djs_loss(
                T=global_mutual_M_R_x2,
                T_prime=global_mutual_M_R_x_prime2,
            )

            global_mutual_loss = (global_mutual_loss1 + global_mutual_loss2) / 2

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'mine_loss': global_mutual_loss,
                   }
            return out

        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'mine_loss': global_mutual_loss,
                   }
            return out
        else:
            raise NotImplementedError


def build_DGTrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and ('DGTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224':
        backbone = deit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224':
        backbone = vit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224':
        backbone = eva02_tiny_patch14_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_distilled_patch16_224':
        backbone = deit_tiny_patch16_224_distill(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_distilled_patch16_224':
        backbone = deit_tiny_patch16_224_distill(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    if (cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224' or cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224' or cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224'
            or cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_distilled_patch16_224'  or cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_distilled_patch16_224'):
        pass
    else:
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = DGTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'DGTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
