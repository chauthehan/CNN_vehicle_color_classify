import argparse
import tensorflow as tf 
from tensorflow.keras.models import load_model
import cv2 
import numpy as np 
from imutils import paths
import time

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', help='path to model')
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())

model = load_model(args['model'])

acc = 0
total = 0

img = cv2.imread(args['image'])
#copy_img = img.copy()
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)
img = img/255.0

tic = time.time()
pred = model.predict(img)
toc = time.time()

pred = np.argmax(pred)
print(pred)
if pred==0:
    print('Predicted color: Black')
if pred==2:
    print('Predicted color: Blue')
if pred==1:
    print('Predicted color: Red')
if pred==4:
    print('Predicted color: Grey')
if pred==5:
    print('Predicted color: Yellow')
if pred==3:
    print('Predicted color: White')
