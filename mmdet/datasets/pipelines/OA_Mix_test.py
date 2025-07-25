
import cv2
import numpy as np
from oa_mix import OAMix
from PIL import Image

def load_yolo_bboxes(txt_file, img_width, img_height):
    bboxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center - height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center + height / 2) * img_height
            bboxes.append([x1, y1, x2, y2])
    return np.array(bboxes, dtype=np.float32)

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image {image_path} could not be loaded.")
    return img

def save_image(image_path, img):
    cv2.imwrite(image_path, img)

def main(image_path, label_path, output_image_path):
    img = load_image(image_path)
    h_img, w_img, _ = img.shape
    bboxes = load_yolo_bboxes(label_path, w_img, h_img)
    oamix = OAMix(version='augmix', num_views=1, keep_orig=False, severity=10, mixture_depth=2)
    results = {'img': img, 'gt_bboxes': bboxes}
    augmented_results = oamix(results)
    augmented_img = augmented_results['img']
    save_image(output_image_path, augmented_img)

import os
from tqdm import tqdm

if __name__ == "__main__":
    image_folder = 'images/'
    label_folder = 'labels/'
    output_folder = 'augmented_images/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    for image_filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_filename)
        label_path = os.path.join(label_folder, image_filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        output_image_path = os.path.join(output_folder, image_filename)
        main(image_path, label_path, output_image_path)



















