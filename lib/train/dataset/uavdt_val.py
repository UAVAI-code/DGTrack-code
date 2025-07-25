
import os
import numpy as np
import torch
import random
from collections import OrderedDict
from lib.train.data import jpeg4py_loader
from lib.test.utils.load_text import load_text

import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings

from lib.test.utils.load_text import load_text

import os
import pandas as pd
import torch
from collections import OrderedDict

class UAVDT_val(BaseVideoDataset):
    def __init__(self, root, image_loader=None, split='val', seq_ids=None, data_fraction=None):
        """
        初始化 UAVDT 数据集
        :param root: 数据集的根目录
        :param image_loader: 图像加载器
        :param split: 数据集划分（train/val/test）
        :param seq_ids: 指定使用哪些序列
        :param data_fraction: 使用数据集的一部分
        """
        self.root = root
        self.split = split
        self.sequence_dir = os.path.join(root, f"sequences_{split}")
        self.anno_dir = os.path.join(root, 'anno')
        self.sequence_list = self._get_sequence_list()

        if seq_ids is not None:
            self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def _get_sequence_list(self):
        """获取所有序列的名称"""
        return sorted([seq for seq in os.listdir(self.sequence_dir) if os.path.isdir(os.path.join(self.sequence_dir, seq))])

    def _read_bb_anno(self, seq_name):
        """读取标注文件"""
        gt_path = os.path.join(self.anno_dir, f"{seq_name}_gt.txt")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"标注文件 {gt_path} 不存在！")
        gt_data = pd.read_csv(gt_path, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'in_view', 'occlusion'])
        return gt_data

    def _read_target_visible(self, seq_data):
        """计算目标的可见性"""
        visible = (seq_data['w'] > 0) & (seq_data['h'] > 0)
        visible_ratio = visible.astype(float)
        return torch.ByteTensor(visible.values), torch.FloatTensor(visible_ratio.values)

    def get_sequence_info(self, seq_name, target_id=1):
        """
        获取指定序列的信息
        :param seq_name: 序列名称
        :param target_id: 指定跟踪目标的 ID（用于单目标跟踪）
        :return: 包括 bbox, valid, visible, visible_ratio 的字典
        """
        gt_data = self._read_bb_anno(seq_name)

        if target_id is not None:
            gt_data = gt_data[gt_data['id'] == target_id]

        valid = (gt_data['w'] > 0) & (gt_data['h'] > 0)
        visible, visible_ratio = self._read_target_visible(gt_data)

        bbox = torch.tensor(gt_data[['x', 'y', 'w', 'h']].values, dtype=torch.float32)
        valid = torch.tensor(valid.values, dtype=torch.bool)

        return {
            'bbox': bbox,
            'valid': valid,
            'visible': visible,
            'visible_ratio': visible_ratio
        }

    def _get_sequence_path(self, seq_name):
        """获取序列文件夹路径"""
        return os.path.join(self.sequence_dir, seq_name)

    def _get_frame_path(self, seq_name, frame_id):
        """获取帧图像的路径"""
        seq_path = self._get_sequence_path(seq_name)
        return os.path.join(seq_path, f"img{frame_id:06d}.jpg")

    def _get_frame(self, seq_name, frame_id):
        """加载帧图像"""
        frame_path = self._get_frame_path(seq_name, frame_id)
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"帧文件 {frame_path} 不存在！")
        return self.image_loader(frame_path)

    def get_frames(self, seq_name, frame_ids, anno=None):
        """
        获取指定帧图像和标注信息
        :param seq_name: 序列名称
        :param frame_ids: 帧 ID 列表
        :param anno: 已加载的标注信息
        :return: 帧图像列表，标注信息字典
        """
        frames = [self._get_frame(seq_name, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_name)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id] for f_id in frame_ids]

        return frames, anno_frames
