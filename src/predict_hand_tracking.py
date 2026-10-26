import cv2
from HandDetector import HandDetector
from ImageClassification import Classifier
import math 
import torch
import joblib
import numpy as np
import torch.nn.functional as F
import cnn_models

# load label binarizer
lb = joblib.load('../outputs/lb.pkl')

model = cnn_models.CustomCNN()
model.load_state_dict(torch.load('../outputs/model.pth'))
print(model)
print('Model loaded')

cap = cv2.VideoCapture(2)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        # imgCropShape = imgCrop.shape
        # imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(w*k)
            try:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                imgWhite = cv2.resize(imgWhite, (224,224))
                image = imgWhite

                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                image = torch.tensor(image, dtype=torch.float)
                image = image.unsqueeze(0)
                
                outputs = model(image)
                _, prediction = torch.max(outputs.data, 1)
                # print(prediction, index)
            except:
                pass

        else:
            k = imgSize/w
            hCal = math.ceil(h*k)
            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                imgWhite = cv2.resize(imgWhite, (224,224))

                image = imgWhite
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                image = torch.tensor(image, dtype=torch.float)
                image = image.unsqueeze(0)
                
                outputs = model(image)
                _, prediction = torch.max(outputs.data, 1)
                # print(prediction, index)
            except:
                pass

        cv2.rectangle(imgOutput, (x-offset,y-offset-50), (x-offset+90,y-offset), (0,255,0), cv2.FILLED)
        cv2.putText(imgOutput, lb.classes_[prediction], (x,y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
        cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset,y+h+offset), (0,255,0), 2)

        if imgCrop is not None and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            cv2.imshow("ImageCrop", imgCrop)
        if imgWhite is not None and imgWhite.shape[0] > 0 and imgWhite.shape[1] > 0:
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    cv2.waitKey(1)
    
 