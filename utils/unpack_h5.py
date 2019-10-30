import os

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

filename = 'city_scape.h5'
use_list = [
    {
        "include_class": ["person", "rider"],
        "class": "human",
        "old": [24, 25],
        "new": [220, 20, 60],
    },
    {
        "include_class": ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'caravan', 'trailer'],
        "class": "vehicle",
        "old": [26, 27, 28, 29, 30, 31, 32, 33],
        "new": [0, 0, 142],
    },
    {
        "include_class": ['traffic sign'],
        "class": "traffic_sign",
        "old": [20],
        "new": [220, 220, 0],
    },
    {
        "include_class": ['road'],
        "class": "road",
        "old": [7],
        "new": [128, 64, 128],
    },
]

output_dirs = filename.replace('.h5', '')
os.makedirs(output_dirs, exist_ok=True)
with h5py.File(filename, 'r') as f:
    all_key = list(f.keys())
    for sub_set in all_key:
        sub_set_dirs = os.path.join(output_dirs, sub_set)
        sub_set_dirs_label = os.path.join(
            output_dirs, '{}_labels'.format(sub_set))
        os.makedirs(sub_set_dirs, exist_ok=True)
        os.makedirs(sub_set_dirs_label, exist_ok=True)
        for index, (sample, label) in tqdm(enumerate(zip(f[sub_set]['x'], f[sub_set]['y'])), desc=sub_set):
            raw_path = os.path.join(sub_set_dirs, '{}.png'.format(index))
            label_path = os.path.join(
                sub_set_dirs_label, '{}.png'.format(index))
            sample = sample.reshape((240, 320, 3))
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            cv2.imwrite(raw_path, sample)
            label = label.reshape((240, 320, 1))
            label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
            rgb_label = np.zeros(label.shape, dtype=np.uint8)
            for each in use_list:
                for i in each['old']:
                    mask = (label == [i, i, i]).all(axis=2)
                    rgb_label[mask] = each['new']
            rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)
            cv2.imwrite(label_path, rgb_label)


df = pd.DataFrame({'name': [each['class'] for each in use_list],
                   'r': [each['new'][0] for each in use_list],
                   'g': [each['new'][1] for each in use_list],
                   'b': [each['new'][2] for each in use_list]})
df.to_csv(os.path.join(output_dirs, 'class_dict.csv'), index=False)
print('Done!')
