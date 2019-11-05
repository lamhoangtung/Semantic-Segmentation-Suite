#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing


# In[2]:


raw_images_path = './apolo_dataset/ColorImage/'  # jpg
raw_label = './apolo_dataset/Label/'  # png
target_size = (320, 240)
output_path = './apolo_dataset/resized/'
num_workers = 32


# In[3]:


output_image_path = os.path.join(output_path, 'images')
output_mask_path = os.path.join(output_path, 'labels')
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_image_path, exist_ok=True)
os.makedirs(output_mask_path, exist_ok=True)


# In[4]:


def view_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[24]:


use_list = [
    {
        "class": "vehicle",
        "old": [1, 33, 161, 34, 162, 35, 163, 36, 164, 37, 165, 38, 166, 39, 167, 40, 168],
        "new": [0, 0, 142],
    },
    {
        "class": "traffic_sign",
        "old": [83],
        "new": [220, 220, 0],
    },
    {
        "class": "road",
        "old": [49],
        "new": [128, 64, 128],
    },
]


# In[31]:


def resize_and_clean_sample(image_file):
    try:
        file_name = os.path.basename(image_file).replace('.jpg', '')
        raw_img = cv2.imread(image_file)
        raw_img = cv2.resize(raw_img, target_size)

        label_path = image_file.replace(
            raw_images_path, raw_label).replace('.jpg', '_bin.png')
        mask_label = cv2.imread(label_path)

        total_pixel = mask_label.shape[0]*mask_label.shape[1]

        # Drop sample have bridge
        mask = (mask_label == [98, 98, 98]).all(axis=2)
        if np.sum(mask) > 0:
            raise Exception('Sample have bridge')

        # Drop sample have more than 1% pixel unlabeled
        mask = (mask_label == [255, 255, 255]).all(axis=2)
        if np.sum(mask) > 0.01*total_pixel:
            raise Exception('Sample have too many unlabel pixel')

        rgb_label = np.zeros(mask_label.shape, dtype=np.uint8)
        for each in use_list:
            for i in each['old']:
                mask = (mask_label == [i, i, i]).all(axis=2)
                if each['class'] == 'road':
                    if np.sum(mask) <= 0.02*total_pixel:
                        raise Exception(
                            'Sample have too little amount of road')
                rgb_label[mask] = each['new']
        rgb_label = cv2.resize(rgb_label, target_size,
                               interpolation=cv2.INTER_LINEAR)
        rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(output_image_path,
                                 '{}.jpg'.format(file_name)), raw_img)
        cv2.imwrite(os.path.join(output_mask_path,
                                 '{}.png'.format(file_name)), rgb_label)
    except Exception as ex:
        print(ex, 'at', image_file)


# In[ ]:


all_records = glob.glob(os.path.join(raw_images_path, 'Record*'))
for index_record, each_record in enumerate(all_records):
    all_cameras = glob.glob(os.path.join(each_record, 'Camera*'))
    for index_cam, each_cam in enumerate(all_cameras):
        print('With record: {}/{}. Camera {}/{}'.format(index_record +
                                                        1, len(all_records), index_cam+1, len(all_cameras)))
        print('- Record path:', each_record)
        print('- Camera path:', each_cam)
        all_images = glob.glob(os.path.join(each_cam, '*.jpg'))
        pool = multiprocessing.Pool(num_workers)
        _ = list(tqdm(pool.imap(resize_and_clean_sample, all_images),
                      total=len(all_images), desc="Cleaning"))
        pool.terminate()
        print('===========================')

print('DONE!')
