import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm

from mmdet.datasets.pipelines.oa_mix1 import OAMix

import time

import sys
import os



current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, project_root)
print(f"Added to path: {project_root}")

print(f"Check paths in sys.path:")
for path in sys.path:
    print(path)


ENVIRONMENT = 'daylight'
NUM_AUG_GROUPS = 1

BASE_DIR = "/home/lzw/LEF/train_data_uavdt"
SEQ_ROOT = os.path.join(BASE_DIR, "uavdt/sequences")
OUTPUT_ROOT = os.path.join(BASE_DIR, "uavdt/sequences_aug")


def read_sequence_indices(env_file_path):
    """读取环境文件的索引列表"""
    with open(env_file_path, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


def read_sequence_names(list_file_path):
    """读取sequences文件夹的列表文件"""
    with open(list_file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_sequence_gt(gt_path):
    """加载groundtruth.txt文件内容"""
    with open(gt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def process_single_image(orig_img, gt_line, oamix):
    """处理单张图像生成多次增强结果，并保存对比图"""


    x, y, w, h = map(float, gt_line.split(','))
    x1, y1, x2, y2 = x, y, x + w, y + h
    bboxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    results = []
    for _ in range(NUM_AUG_GROUPS):
        res = {'img': img_rgb.copy(), 'gt_bboxes': bboxes}
        augmented = oamix(res)
        results.append(cv2.cvtColor(augmented['img'], cv2.COLOR_RGB2BGR))
    return results






def process_sequence(sequence_path):
    """处理单个视频序列"""
    seq_name = os.path.basename(sequence_path)
    print(f"\nProcessing sequence: {seq_name}")

    gt_path = os.path.join(sequence_path, "groundtruth.txt")
    gt_lines = load_sequence_gt(gt_path)

    oamix = OAMix(


        version='augmix.all',
        num_views=1,
        keep_orig=False,
        severity=9,
        mixture_width=4,
        mixture_depth=4,
        random_box_scale=(0.005, 0.02),
        random_box_ratio=(3, 1 / 3),
        oa_random_box_scale=(0.002, 0.03),
        oa_random_box_ratio=(3, 1 / 3),
        num_bboxes=(7, 15),
        spatial_ratio=2,
        sigma_ratio=0.5,

    )



    output_folders = []
    for aug_idx in range(NUM_AUG_GROUPS):
        if NUM_AUG_GROUPS > 1:
            folder_name = f"{seq_name}_aug1{aug_idx + 1}"
        else:
            folder_name = f"{seq_name}_aug1"

        output_dir = os.path.join(OUTPUT_ROOT, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        output_folders.append(output_dir)

        with open(os.path.join(output_dir, "groundtruth.txt"), 'w') as f:
            f.write("\n".join(gt_lines))

        src_meta = os.path.join(sequence_path, "meta.json")
        if os.path.exists(src_meta):
            try:
                with open(src_meta, 'r') as f:
                    meta = json.load(f)

                for tag in meta.get('tags', []):
                    if tag.get('name') == 'sequence':
                        tag['value'] = folder_name

                with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
                    json.dump(meta, f, indent=4)
            except Exception as e:
                print(f"处理 {seq_name} 的 meta.json 时出错: {e}")
        else:
            print(f"警告：{seq_name} 缺少 meta.json")

        src_att = os.path.join(sequence_path, f"{seq_name}_att.txt")
        if os.path.exists(src_att):
            dest_att = os.path.join(output_dir, f"{folder_name}_att.txt")
            shutil.copyfile(src_att, dest_att)
        else:
            print(f"警告：{seq_name} 缺少 {seq_name}_att.txt")

    frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith(('.jpg', '.png'))])
    for frame_idx, frame_name in enumerate(tqdm(frame_files, desc="Frames")):
        frame_path = os.path.join(sequence_path, frame_name)
        orig_img = cv2.imread(frame_path)
        if orig_img is None:
            print(f"Warning: 跳过无法读取的帧 {frame_path}")
            continue
        try:
            enhanced_images = process_single_image(orig_img, gt_lines[frame_idx], oamix)
        except Exception as e:
            print(f"Error processing {frame_name}: {str(e)}")
            continue
        for aug_idx, img in enumerate(enhanced_images):
            output_path = os.path.join(output_folders[aug_idx], frame_name)
            cv2.imwrite(output_path, img)




def main():
    env_file = os.path.join(BASE_DIR, "uavdt", f"uavdt_{ENVIRONMENT}.txt")
    list_file = os.path.join(SEQ_ROOT, "list.txt")

    selected_indices = read_sequence_indices(env_file)
    all_sequences = read_sequence_names(list_file)
    selected_sequences = [all_sequences[i] for i in selected_indices if i < len(all_sequences)]

    for seq_name in tqdm(selected_sequences, desc=f"Processing {ENVIRONMENT} sequences"):
        seq_path = os.path.join(SEQ_ROOT, seq_name)
        if not os.path.exists(seq_path):
            print(f"Warning: 跳过不存在的序列 {seq_name}")
            continue
        process_sequence(seq_path)


if __name__ == "__main__":
    main()
