# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:42:00 2020
@author: aneys
"""

#This piece of code loads data from the two numpy files and trains the model.
#The CNN model is then saved.

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
from google.colab import drive
#drive.mount('/content/drive')
 
file_path_X = '/content/drive/My Drive/CoLab_data/X.npy'
file_path_y = '/content/drive/My Drive/CoLab_data/y.npy'
 
X = np.load(file_path_X)
y = np.load(file_path_y)
 
X = X.reshape(X.shape[0],480,640,1)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

del X
del y
del X_test
del y_test

print('program successfully completed.')



model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=X_train.shape[1:])) 
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

print('program successfully completed.')



epochs = 500
batch_size = 50
hist = History()

checkpointer = ModelCheckpoint(filepath='/content/drive/My Drive/CoLab_data/checkpoint1.hdf5', verbose=1, save_best_only=True, monitor='val_loss', mode='min')

# Complie Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()

model_fit = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, hist], verbose=1)
model.save('/content/drive/My Drive/CoLab_data/trained_model.h5')
print('program successfully completed.')
        



  

