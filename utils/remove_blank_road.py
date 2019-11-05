import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

all_data_path = './apolo_city_scape/train/'
all_mask_path = './apolo_city_scape/train_labels/'
total_pixel = 76800
count = 0

for img_path in tqdm(glob.glob(os.path.join(all_data_path, '*.jpg'))):
    mask_path = img_path.replace(all_data_path, all_mask_path).replace('.jpg', '.png')
    mask = cv2.imread(mask_path)
    # print(mask_path)
    # Drop sample have no road
    temp_mask = (mask == [128, 64, 128]).all(axis=2)
    if np.sum(temp_mask) <= 0.05 * total_pixel:
    #     img = cv2.imread(img_path)
    #     cv2.imshow('img', img)
    #     cv2.imshow('msk', mask)
    #     cv2.waitKey(0)
        count += 1
        os.remove(img_path)
        os.remove(mask_path)

print(count)