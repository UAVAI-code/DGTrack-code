import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class GridHead(BaseModule):

    def __init__(self,
                 grid_points=9,
                 num_convs=8,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 point_feat_channels=64,
                 deconv_kernel_size=4,
                 class_agnostic=False,
                 loss_grid=dict(
                     type='CrossEntropyLoss', use_sigmoid=True,
                     loss_weight=15),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=36),
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d', 'Linear']),
                     dict(
                         type='Normal',
                         layer='ConvTranspose2d',
                         std=0.001,
                         override=dict(
                             type='Normal',
                             name='deconv2',
                             std=0.001,
                             bias=-np.log(0.99 / 0.01)))
                 ]):
        super(GridHead, self).__init__(init_cfg)
        self.grid_points = grid_points
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.point_feat_channels = point_feat_channels
        self.conv_out_channels = self.point_feat_channels * self.grid_points
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if isinstance(norm_cfg, dict) and norm_cfg['type'] == 'GN':
            assert self.conv_out_channels % norm_cfg['num_groups'] == 0

        assert self.grid_points >= 4
        self.grid_size = int(np.sqrt(self.grid_points))
        if self.grid_size * self.grid_size != self.grid_points:
            raise ValueError('grid_points must be a square number')

        if not isinstance(self.roi_feat_size, int):
            raise ValueError('Only square RoIs are supporeted in Grid R-CNN')
        self.whole_map_size = self.roi_feat_size * 4

        self.sub_regions = self.calc_sub_regions()

        self.convs = []
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            stride = 2 if i == 0 else 1
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=True))
        self.convs = nn.Sequential(*self.convs)

        self.deconv1 = nn.ConvTranspose2d(
            self.conv_out_channels,
            self.conv_out_channels,
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=grid_points)
        self.norm1 = nn.GroupNorm(grid_points, self.conv_out_channels)
        self.deconv2 = nn.ConvTranspose2d(
            self.conv_out_channels,
            grid_points,
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=grid_points)

        self.neighbor_points = []
        grid_size = self.grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                neighbors = []
                if i > 0:
                    neighbors.append((i - 1) * grid_size + j)
                if j > 0:
                    neighbors.append(i * grid_size + j - 1)
                if j < grid_size - 1:
                    neighbors.append(i * grid_size + j + 1)
                if i < grid_size - 1:
                    neighbors.append((i + 1) * grid_size + j)
                self.neighbor_points.append(tuple(neighbors))
        self.num_edges = sum([len(p) for p in self.neighbor_points])

        self.forder_trans = nn.ModuleList()
        self.sorder_trans = nn.ModuleList()
        for neighbors in self.neighbor_points:
            fo_trans = nn.ModuleList()
            so_trans = nn.ModuleList()
            for _ in range(len(neighbors)):
                fo_trans.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.point_feat_channels,
                            self.point_feat_channels,
                            5,
                            stride=1,
                            padding=2,
                            groups=self.point_feat_channels),
                        nn.Conv2d(self.point_feat_channels,
                                  self.point_feat_channels, 1)))
                so_trans.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.point_feat_channels,
                            self.point_feat_channels,
                            5,
                            1,
                            2,
                            groups=self.point_feat_channels),
                        nn.Conv2d(self.point_feat_channels,
                                  self.point_feat_channels, 1)))
            self.forder_trans.append(fo_trans)
            self.sorder_trans.append(so_trans)

        self.loss_grid = build_loss(loss_grid)

    def forward(self, x):
        assert x.shape[-1] == x.shape[-2] == self.roi_feat_size
        x = self.convs(x)

        c = self.point_feat_channels
        x_fo = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_fo[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_fo[i] = x_fo[i] + self.forder_trans[i][j](
                    x[:, point_idx * c:(point_idx + 1) * c])

        x_so = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_so[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_so[i] = x_so[i] + self.sorder_trans[i][j](x_fo[point_idx])

        x2 = torch.cat(x_so, dim=1)
        x2 = self.deconv1(x2)
        x2 = F.relu(self.norm1(x2), inplace=True)
        heatmap = self.deconv2(x2)

        if self.training:
            x1 = x
            x1 = self.deconv1(x1)
            x1 = F.relu(self.norm1(x1), inplace=True)
            heatmap_unfused = self.deconv2(x1)
        else:
            heatmap_unfused = heatmap

        return dict(fused=heatmap, unfused=heatmap_unfused)

    def calc_sub_regions(self):
        """Compute point specific representation regions.

        See Grid R-CNN Plus (https://arxiv.org/abs/1906.05688) for details.
        """
        half_size = self.whole_map_size // 4 * 2
        sub_regions = []
        for i in range(self.grid_points):
            x_idx = i // self.grid_size
            y_idx = i % self.grid_size
            if x_idx == 0:
                sub_x1 = 0
            elif x_idx == self.grid_size - 1:
                sub_x1 = half_size
            else:
                ratio = x_idx / (self.grid_size - 1) - 0.25
                sub_x1 = max(int(ratio * self.whole_map_size), 0)

            if y_idx == 0:
                sub_y1 = 0
            elif y_idx == self.grid_size - 1:
                sub_y1 = half_size
            else:
                ratio = y_idx / (self.grid_size - 1) - 0.25
                sub_y1 = max(int(ratio * self.whole_map_size), 0)
            sub_regions.append(
                (sub_x1, sub_y1, sub_x1 + half_size, sub_y1 + half_size))
        return sub_regions

    def get_targets(self, sampling_results, rcnn_train_cfg):
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results],
                               dim=0).cpu()
        pos_gt_bboxes = torch.cat(
            [res.pos_gt_bboxes for res in sampling_results], dim=0).cpu()
        assert pos_bboxes.shape == pos_gt_bboxes.shape

        x1 = pos_bboxes[:, 0] - (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
        y1 = pos_bboxes[:, 1] - (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
        x2 = pos_bboxes[:, 2] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
        y2 = pos_bboxes[:, 3] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
        pos_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        pos_bbox_ws = (pos_bboxes[:, 2] - pos_bboxes[:, 0]).unsqueeze(-1)
        pos_bbox_hs = (pos_bboxes[:, 3] - pos_bboxes[:, 1]).unsqueeze(-1)

        num_rois = pos_bboxes.shape[0]
        map_size = self.whole_map_size
        targets = torch.zeros((num_rois, self.grid_points, map_size, map_size),
                              dtype=torch.float)

        factors = []
        for j in range(self.grid_points):
            x_idx = j // self.grid_size
            y_idx = j % self.grid_size
            factors.append((1 - x_idx / (self.grid_size - 1),
                            1 - y_idx / (self.grid_size - 1)))

        radius = rcnn_train_cfg.pos_radius
        radius2 = radius**2
        for i in range(num_rois):
            if (pos_bbox_ws[i] <= self.grid_size
                    or pos_bbox_hs[i] <= self.grid_size):
                continue
            for j in range(self.grid_points):
                factor_x, factor_y = factors[j]
                gridpoint_x = factor_x * pos_gt_bboxes[i, 0] + (
                    1 - factor_x) * pos_gt_bboxes[i, 2]
                gridpoint_y = factor_y * pos_gt_bboxes[i, 1] + (
                    1 - factor_y) * pos_gt_bboxes[i, 3]

                cx = int((gridpoint_x - pos_bboxes[i, 0]) / pos_bbox_ws[i] *
                         map_size)
                cy = int((gridpoint_y - pos_bboxes[i, 1]) / pos_bbox_hs[i] *
                         map_size)

                for x in range(cx - radius, cx + radius + 1):
                    for y in range(cy - radius, cy + radius + 1):
                        if x >= 0 and x < map_size and y >= 0 and y < map_size:
                            if (x - cx)**2 + (y - cy)**2 <= radius2:
                                targets[i, j, y, x] = 1
        sub_targets = []
        for i in range(self.grid_points):
            sub_x1, sub_y1, sub_x2, sub_y2 = self.sub_regions[i]
            sub_targets.append(targets[:, [i], sub_y1:sub_y2, sub_x1:sub_x2])
        sub_targets = torch.cat(sub_targets, dim=1)
        sub_targets = sub_targets.to(sampling_results[0].pos_bboxes.device)
        return sub_targets

    def loss(self, grid_pred, grid_targets):
        loss_fused = self.loss_grid(grid_pred['fused'], grid_targets)
        loss_unfused = self.loss_grid(grid_pred['unfused'], grid_targets)
        loss_grid = loss_fused + loss_unfused
        return dict(loss_grid=loss_grid)

    def get_bboxes(self, det_bboxes, grid_pred, img_metas):
        assert det_bboxes.shape[0] == grid_pred.shape[0]
        det_bboxes = det_bboxes.cpu()
        cls_scores = det_bboxes[:, [4]]
        det_bboxes = det_bboxes[:, :4]
        grid_pred = grid_pred.sigmoid().cpu()

        R, c, h, w = grid_pred.shape
        half_size = self.whole_map_size // 4 * 2
        assert h == w == half_size
        assert c == self.grid_points

        grid_pred = grid_pred.view(R * c, h * w)
        pred_scores, pred_position = grid_pred.max(dim=1)
        xs = pred_position % w
        ys = pred_position // w

        for i in range(self.grid_points):
            xs[i::self.grid_points] += self.sub_regions[i][0]
            ys[i::self.grid_points] += self.sub_regions[i][1]

        pred_scores, xs, ys = tuple(
            map(lambda x: x.view(R, c), [pred_scores, xs, ys]))

        widths = (det_bboxes[:, 2] - det_bboxes[:, 0]).unsqueeze(-1)
        heights = (det_bboxes[:, 3] - det_bboxes[:, 1]).unsqueeze(-1)
        x1 = (det_bboxes[:, 0, None] - widths / 2)
        y1 = (det_bboxes[:, 1, None] - heights / 2)
        abs_xs = (xs.float() + 0.5) / w * widths + x1
        abs_ys = (ys.float() + 0.5) / h * heights + y1

        x1_inds = [i for i in range(self.grid_size)]
        y1_inds = [i * self.grid_size for i in range(self.grid_size)]
        x2_inds = [
            self.grid_points - self.grid_size + i
            for i in range(self.grid_size)
        ]
        y2_inds = [(i + 1) * self.grid_size - 1 for i in range(self.grid_size)]

        bboxes_x1 = (abs_xs[:, x1_inds] * pred_scores[:, x1_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, x1_inds].sum(dim=1, keepdim=True))
        bboxes_y1 = (abs_ys[:, y1_inds] * pred_scores[:, y1_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, y1_inds].sum(dim=1, keepdim=True))
        bboxes_x2 = (abs_xs[:, x2_inds] * pred_scores[:, x2_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, x2_inds].sum(dim=1, keepdim=True))
        bboxes_y2 = (abs_ys[:, y2_inds] * pred_scores[:, y2_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, y2_inds].sum(dim=1, keepdim=True))

        bbox_res = torch.cat(
            [bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2, cls_scores], dim=1)
        bbox_res[:, [0, 2]].clamp_(min=0, max=img_metas[0]['img_shape'][1])
        bbox_res[:, [1, 3]].clamp_(min=0, max=img_metas[0]['img_shape'][0])

        return bbox_res
