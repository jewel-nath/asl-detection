# Adapted from "American Sign Language Recognition using Deep Learning" by Sovit Ranjan Rath, DebuggerCafe, 2020
# (https://debuggercafe.com/american-sign-language-recognition-using-deep-learning/).
'''
USAGE:
python src/predict_image.py --img A_test.jpg
'''

import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import argparse
import albumentations
import torch.nn.functional as F
import time
import os
import cnn_models

ROOT_PATH = os.path.realpath(os.path.join(__file__, '..', '..'))

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img', default='A-test-1.jpg', type=str,
    help='path for the image to test on')
args = vars(parser.parse_args())

# load label binarizer
lb = joblib.load(os.path.join(ROOT_PATH, 'assets', 'models', 'lb.pkl'))

aug = albumentations.Compose([
                albumentations.Resize(224, 224, always_apply=True),
])

# model = models.MobineNetV2(pretrained=False, requires_grad=False)
model = cnn_models.CustomCNN()
model.load_state_dict(torch.load(os.path.join(ROOT_PATH, 'assets', 'models', 'model.pth')))
print(model)
print('Model loaded')

image = cv2.imread(os.path.join(ROOT_PATH, 'data', 'test', args['img']))
image_copy = image.copy()
 
image = aug(image=np.array(image))['image']
image = np.transpose(image, (2, 0, 1)).astype(np.float32)
image = torch.tensor(image, dtype=torch.float)
image = image.unsqueeze(0)
print(image.shape)
 
start = time.time()
outputs = model(image)
_, preds = torch.max(outputs.data, 1)
print('PREDS', preds)
print(f"Predicted output: {lb.classes_[preds]}")
end = time.time()
print(f"{(end-start):.3f} seconds")
 
cv2.putText(image_copy, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow('image', image_copy)
os.makedirs(os.path.join(ROOT_PATH, 'outputs'), exist_ok=True)
cv2.imwrite(os.path.join(ROOT_PATH, 'outputs', args['img']), image_copy)
cv2.waitKey(0)