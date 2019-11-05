import cv2
import os
import glob
from tqdm import tqdm

dataset_dir = 'apolo_city_scape'
view_mode = 1

train_input_names = []
train_output_names = []
val_input_names = []
val_output_names = []
test_input_names = []
test_output_names = []
for file in os.listdir(dataset_dir + "/train"):
    cwd = os.getcwd()
    train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
for file in os.listdir(dataset_dir + "/train_labels"):
    cwd = os.getcwd()
    train_output_names.append(
        cwd + "/" + dataset_dir + "/train_labels/" + file)
for file in os.listdir(dataset_dir + "/val"):
    cwd = os.getcwd()
    val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
for file in os.listdir(dataset_dir + "/val_labels"):
    cwd = os.getcwd()
    val_output_names.append(
        cwd + "/" + dataset_dir + "/val_labels/" + file)
for file in os.listdir(dataset_dir + "/test"):
    cwd = os.getcwd()
    test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
for file in os.listdir(dataset_dir + "/test_labels"):
    cwd = os.getcwd()
    test_output_names.append(
        cwd + "/" + dataset_dir + "/test_labels/" + file)
train_input_names.sort(), train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()

for img, mask in zip(train_input_names, train_output_names):
    cv2.imshow('img', cv2.imread(img))
    cv2.imshow('mask', cv2.imread(mask))
    cv2.waitKey(view_mode)

for img, mask in zip(val_input_names, val_output_names):
    cv2.imshow('img', cv2.imread(img))
    cv2.imshow('mask', cv2.imread(mask))
    cv2.waitKey(view_mode)
