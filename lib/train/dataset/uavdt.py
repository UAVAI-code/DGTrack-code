
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


class UAVDT(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not  the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().uavdt_dir if root is None else root
        super().__init__('UAVDT', root, image_loader)

        self.sequence_list = self._get_sequence_list()

        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')




            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'daylight':
                file_path = os.path.join(ltr_path, 'data_specs', 'uavdt_daylight_split.txt')
            elif split == 'night':
                file_path = os.path.join(ltr_path, 'data_specs', 'uavdt_night_split.txt')
            elif split == 'fog':
                file_path = os.path.join(ltr_path, 'data_specs', 'uavdt_fog_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'uavdt_val_split.txt')

            elif split == 'daylight_aug1':
                file_path = os.path.join(ltr_path, 'data_specs', 'uavdt_daylight_aug1_split.txt')

            elif split == 'daylight_aug2':
                file_path = os.path.join(ltr_path, 'data_specs', 'uavdt_daylight_aug2_split.txt')



            else:
                raise ValueError('Unknown split name.')
            seq_ids = pandas.read_csv(file_path, header=None, dtype=np.int64).squeeze("columns").values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))



        self.sequence_list = [self.sequence_list[i] for i in seq_ids]
        print(self.sequence_list)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))


        self.sequence_meta_info = self._load_meta_info()

        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'uavdt'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta.json')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': 'vehicle',
                                       'motion_class': None,
                                       'major_class':None,
                                       'root_class': None,
                                       'motion_adverb':  None})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = [line.strip() for line in f]
        print(dir_list)
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        self.gt = gt
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        """
        根据UAVDT序列的属性文件生成target_visible和visible_ratio
        :param seq_path: 序列路径，包含att文件
        :return: target_visible, visible_ratio
        """

        num_frames = self.gt.shape[0]



        absence = np.array([1] * num_frames)
        cover = np.array([8] * num_frames)


        absence_tensor = torch.ByteTensor(absence)
        cover_tensor = torch.ByteTensor(cover)

        target_visible = absence_tensor & (cover_tensor > 0).byte()
        visible_ratio = cover_tensor.float() / 8

        return target_visible, visible_ratio


    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img{:06}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta
