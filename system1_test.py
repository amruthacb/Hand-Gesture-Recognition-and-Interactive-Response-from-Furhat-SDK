# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:33:43 2020

@author: aneys
"""

import csv
import numpy as np
import cv2
import os, shutil
from matplotlib import image
from matplotlib import pyplot
from PIL import Image
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Activation
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, History

with open("Video Data_new.csv", 'r') as f:
     points = list(csv.reader(f, delimiter=","))  

all_data = np.array(points[0:], dtype='str')
source_all = np.array(all_data[1:,1])
cam_facing_side_all = np.array(all_data[1:,3])
gesture_all = np.array(all_data[1:,4])
ID_all = np.array(all_data[1:,0],dtype = 'i')
frame_all = np.array(all_data[1:,2],dtype = 'i')
coordinates_all = np.array(all_data[1:,5:],dtype = 'float') 

folder = "Frames_test"
count = 0

for image in os.listdir(folder):
    
    image_path = 'Frames_test'+'/'+image    
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    img_x,img_y = img.shape
    
   
    if ((img_x != 480) & (img_y != 640)):       
       continue
     
    
    X_temp=np.array(np.array(img), dtype = np.uint8)
    X_temp = X_temp.reshape(1,480,640)
    names = image.strip('.png')
    segmented_name = names.split('_')
    id_video_temp= int(segmented_name[0])    
    cam_facing_temp = segmented_name[1]      
    gesture_temp  = segmented_name[2]    
    frame_no_temp = int(segmented_name[3])      

    y_temp = coordinates_all[(ID_all == id_video_temp) & (cam_facing_side_all == cam_facing_temp) & (gesture_all == gesture_temp) & (frame_all == frame_no_temp) ]    
    y_temp = np.array(y_temp, dtype = np.int)
    
    if count == 0:
        y_test = y_temp
        X_test = X_temp
    else:
        y_test = np.append(y_test,y_temp, axis = 0)
        X_test = np.append(X_test,X_temp,axis = 0)
         
    count = count + 1   
    
       
X_test = X_test.reshape(X_test.shape[0],480,640,1)
    
model = Sequential()

#model.add(BatchNormalization())
#model.add(Conv2D(48, (3,3), padding='same', activation='relu' ))
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=X_test.shape[1:])) # Input shape: (96, 96, 1)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))


model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))


# Convert all values to 1D array
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(80,activation='relu'))

model.load_weights('checkpoint1.hdf5')    

y_predict = model.predict(X_test)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Finding Mean square error per image and for the batch
text_Stat, test_acc = model.evaluate(X_test, y_test)
MSE_individual = np.array(tf.keras.losses.MSE(y_test, y_predict), dtype = np.float)
# ************************************************************

color1 = (255, 0, 0)
color2 = (0, 0, 255)
radious = 5
thickness = 1
img_no = 0
for image in os.listdir(folder):
    
    image_path = 'Frames_test'+'/'+image    
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    img_x,img_y = img.shape
    
   
    if ((img_x != 480) & (img_y != 640)):
       #print(image)
       #print(img_x,img_y)
       continue
   
    for target in np.arange(0,79,2):
        #print(target)
        x_coordinate=np.array(y_predict[img_no,target])
        y_coordinate=np.array(y_predict[img_no,target+1])
        cv2.circle(img,(x_coordinate, y_coordinate), radious,color1,-1)
        
    for target in np.arange(0,79,2):
        #print(target)
        x_coordinate=np.array(y_test[img_no,target])
        y_coordinate=np.array(y_test[img_no,target+1])
        cv2.circle(img,(x_coordinate, y_coordinate), radious,color2,-1)
    
    
    path = "Frames_test_op/"+image      
    cv2.imwrite(path, img)
    img_no = img_no + 1;



    
    
    
    
    
    
    
    
    
    
    
    
  
