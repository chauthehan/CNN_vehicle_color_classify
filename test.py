import argparse
import tensorflow as tf 
from tensorflow.keras.models import load_model
import cv2 
import numpy as np 
from imutils import paths
import time

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', help='path to model')
args = vars(ap.parse_args())

model = load_model(args['model'])

acc = 0
total = 0
for path in paths.list_images('test'):
    if 'Black' in path: #ok
        gt = 0
    if 'Blue' in path:
        gt = 2
    if 'Red' in path:#ok
        gt = 1
    if 'White' in path:#ok
        gt = 3
    if 'grey' in path: #ok
        gt = 4
    if 'yellow' in path: #ok
        gt = 5
    
    
    img = cv2.imread(path)
    #copy_img = img.copy()
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img/255.0

    tic = time.time()
    pred = model.predict(img)

    toc = time.time()

    pred = np.argmax(pred)
    #print(pred)
    if pred==gt:
        acc += 1
    total += 1
    #cv2.imshow('',copy_img)
   # cv2.waitKey(0)
print('acc: ', acc)
print('total: ', total)
print('Accuracy: ', acc/total)
print('Sec per img: ', toc-tic)