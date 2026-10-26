# Adapted from "American Sign Language Recognition using Deep Learning" by Sovit Ranjan Rath, DebuggerCafe, 2020
# (https://debuggercafe.com/american-sign-language-recognition-using-deep-learning/).
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os

ROOT_PATH = os.path.realpath(os.path.join(__file__, '..', '..'))

from torchvision import models

# load the binarized labels
print('Loading label binarizer...')
lb = joblib.load(os.path.join(ROOT_PATH, 'assets', 'models', 'lb.pkl'))

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(lb.classes_))

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x