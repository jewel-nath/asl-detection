# Adapted from "American Sign Language Recognition using Deep Learning" by Sovit Ranjan Rath, DebuggerCafe, 2020
# (https://debuggercafe.com/american-sign-language-recognition-using-deep-learning/).
import pandas as pd
import numpy as np
import os
import joblib

ROOT_PATH = os.path.realpath(os.path.join(__file__, '..', '..'))

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from imutils import paths

# get all the image paths
image_paths = list(paths.list_images(os.path.join(ROOT_PATH, 'data', 'preprocessed-image')))

# create a DataFrame 
data = pd.DataFrame()

labels = []
for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
    label = image_path.split(os.path.sep)[-2]
    # save the relative path for mapping image to target
    data.loc[i, 'image_path'] = image_path

    labels.append(label)

labels = np.array(labels)
# one hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print(f"The first one hot encoded labels: {labels[0]}")
print(f"Mapping the first one hot encoded label to its category: {lb.classes_[0]}")
print(f"Total instances: {len(labels)}")

for i in range(len(labels)):
    index = np.argmax(labels[i])
    data.loc[i, 'target'] = int(index)

# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# save as CSV file
data.to_csv(os.path.join(ROOT_PATH, 'data', 'data.csv'), index=False)

# pickle the binarized labels
print('Saving the binarized labels as pickled file')
joblib.dump(lb, os.path.join(ROOT_PATH, 'assets', 'models', 'lb.pkl'))

print(data.head(10))