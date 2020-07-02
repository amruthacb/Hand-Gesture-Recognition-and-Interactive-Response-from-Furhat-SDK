# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:53:37 2020
@author: aneys
"""

#This is the code where we put all the image frames into a single folder 'Frames'
#which will be used as input data to create X_train and y_train for the CNN model
#Importing Modules
import csv
import numpy as np
import cv2
import os, shutil

#Create a folder named 'Frames' in the working directory
folder = 'Frames'

#Ensure that the folder is cleaned everytime
#Run this part only if you want to delete the existing files inside 'Frames'
# for filename in os.listdir(folder):
#     file_path = os.path.join(folder, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))
        
#change the video file name for the file from which you want to extract coordinates and frames        
video_file_name = "441/open_dorsal"        
with open("Video Data_new.csv", 'r') as f:
    points = list(csv.reader(f, delimiter=","))  
    
all_data = np.array(points[0:], dtype='str')        
all_headers = all_data[0,:]
ID_all = np.array(all_data[1:,0],dtype = 'i')
source_all = np.array(all_data[1:,1])
frame_all = np.array(all_data[1:,2],dtype = 'i')
cam_facing_side_all = np.array(all_data[1:,3])
gesture_all = np.array(all_data[1:,4])
coordinates_all = np.array(all_data[1:,5:],dtype = 'float') 

frame_no = frame_all[source_all == video_file_name+".webm"]
cam_facing_side = cam_facing_side_all[source_all == video_file_name+".webm"]
gesture = gesture_all[source_all == video_file_name+".webm"]
coordinates = coordinates_all[source_all == video_file_name+".webm",:]
ID_no = ID_all[source_all == video_file_name+".webm"]


video_reader = cv2.VideoCapture(video_file_name + ".webm")
ret, frame = video_reader.read()
counter = 0
while ret:
    ret, frame = video_reader.read()
    if not ret:
        break
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cam_facing_side_frame = cam_facing_side[frame_no==counter]
    gesture_frame = gesture[frame_no==counter]
    Used_ID = ID_no[frame_no == counter]   
    namefile = str(Used_ID)+"_"+str(cam_facing_side_frame)+"_"+str(gesture_frame)+"_"+str(counter)
    namefile = namefile.replace('[','').replace(']','').replace("'",'')    
    path = "Frames/"+namefile+".png"      
    cv2.imwrite(path, grayImage)
    counter += 1
video_reader.release()
tot_count = counter



