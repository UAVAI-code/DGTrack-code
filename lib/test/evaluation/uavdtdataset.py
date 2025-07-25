import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class UAVDTDataset(BaseDataset):

    def __init__(self):
        super().__init__()

        self.base_path = os.path.join(self.env_settings.uavdt_path)
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        if self.dataset_name == 'uavdt':
            anno_path = '{}/{}/{}_gt.txt'.format(self.base_path, 'anno', sequence_name)
            frames_path = '{}/{}/{}'.format(self.base_path, 'sequences', sequence_name)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[3:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        return Sequence(sequence_name, frames_list, self.dataset_name, ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split='night'):
        if split in ['daylight', 'night', 'fog']:
            list_file = os.path.join(self.base_path, 'sequences', 'list.txt')
            self.dataset_name = 'uavdt'
        else:
            raise ValueError(f"Unknown split: {split}")

        with open(list_file, 'r') as f:
            all_sequences = [line.strip() for line in f]

        if split is None:
            return all_sequences

        split_file_map = {
            'daylight': 'uavdt_daylight_split.txt',
            'fog': 'uavdt_fog_split.txt',
            'night': 'uavdt_night_split.txt',
        }

        if split not in split_file_map:
            raise ValueError(f"Unknown split: {split}")

        if split in ['daylight', 'night', 'fog']:
            split_file = os.path.join(os.path.dirname(__file__), split_file_map[split])

        with open(split_file, 'r') as f:
            indices = [int(line.strip()) for line in f]

        seqs = [all_sequences[i] for i in indices]

        return seqs