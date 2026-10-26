# Adapted from "American Sign Language Recognition using Deep Learning" by Sovit Ranjan Rath, DebuggerCafe, 2020
# (https://debuggercafe.com/american-sign-language-recognition-using-deep-learning/).
'''
USAGE:
python src/preprocess_image.py --num-images 1200
'''

import pandas as pd
import os
import cv2
import random
import albumentations
import numpy as np
import argparse

ROOT_PATH = os.path.realpath(os.path.join(__file__, '..', '..'))
DATA_PATH = os.path.join(ROOT_PATH, 'data')

from imutils import paths
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-images', default=800, type=int,
    help='number of images to preprocess for each category')
args = vars(parser.parse_args())

print(f"Preprocessing {args['num_images']} from each category...")

# get all the image paths
image_paths = list(paths.list_images(os.path.join(DATA_PATH, 'train')))
dir_paths = os.listdir(os.path.join(DATA_PATH, 'train'))
dir_paths.sort()

root_path = os.path.join(DATA_PATH, 'train')

# get 1000 images from each category
for idx, dir_path  in tqdm(enumerate(dir_paths), total=len(dir_paths)):
    all_images = os.listdir(f"{root_path}/{dir_path}")
    os.makedirs(os.path.join(DATA_PATH, 'preprocessed-image', dir_path), exist_ok=True)
    for i in range(args['num_images']): # how many images to preprocess for each category
        # generate a random id between 0 and 500
        rand_id = (random.randint(0, 800))
        image = cv2.imread(f"{root_path}/{dir_path}/{all_images[rand_id]}")
        image = cv2.resize(image, (224, 224))

        cv2.imwrite(os.path.join(DATA_PATH, 'preprocessed-image', dir_path, f"{dir_path}{i}.jpg"), image)

print('DONE')