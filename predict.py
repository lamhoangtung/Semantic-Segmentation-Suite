import argparse
import glob
import math
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from builders import model_builder
from utils import helpers, utils

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='./simulator/',
                    required=False, help='The path to image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default='./weights/latest_model_MobileUNet-Skip_city_scape.ckpt', required=False,
                    help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=224,
                    help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=320,
                    help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default='MobileUNet-Skip',
                    required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="city_scape",
                    required=False, help='The dataset you are using')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(
    os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image Path -->", args.image_path)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

net_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
net_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                       num_classes=num_classes,
                                       crop_width=args.crop_width,
                                       crop_height=args.crop_height,
                                       is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver = tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

for each_image in (glob.glob(os.path.join(args.image_path, '*.jpg'))):
    # print("Testing image " + args.image)
    start_time = time.time()

    loaded_image = utils.load_image(each_image)
    resized_image = cv2.resize(
        loaded_image, (args.crop_width, args.crop_height))
    input_image = np.expand_dims(np.float32(
        resized_image[:args.crop_height, :args.crop_width]), axis=0)/255.0

    st = time.time()
    output_image = sess.run(network, feed_dict={net_input: input_image})
    print("FPS: {}".format(1/(time.time()-start_time)), end='\r')

    run_time = time.time()-st

    output_image = np.array(output_image[0, :, :, :])
    output_image = helpers.reverse_one_hot(output_image)

    out_vis_image = helpers.colour_code_segmentation(
        output_image, label_values)
    file_name = utils.filepath_to_name(each_image)
    cv2.imwrite("./result/%s_pred.png" % (file_name), cv2.cvtColor(
        np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

# print("Finished!")
# print("Wrote image " + "%s_pred.png" % (file_name))
