import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

all_data_path = './apolo_city_scape_no_vehicle/test/'
all_mask_path = './apolo_city_scape_no_vehicle/test_labels/'

for img_path in tqdm(glob.glob(os.path.join(all_data_path, '*.png'))):
    mask_path = img_path.replace(
        all_data_path, all_mask_path).replace('.jpg', '.png')
    mask = cv2.imread(mask_path)
    mask[(mask == [142, 0, 0]).all(axis=2)] = [0, 0, 0]
    cv2.imwrite(mask_path, mask)
