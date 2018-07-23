#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:26:25 2018

@author: ddeng
"""
import numpy as np
import pandas as pd
import os
import os.path as path
from tqdm import tqdm
import matplotlib.pyplot as plt
def angle_difference(coor1, coor2):
    L1 = np.sqrt(coor1.dot(coor1))
    L2 = np.sqrt(coor2.dot(coor2))
    
    cosine = coor1.dot(coor2)/(L1*L2)
    angle = np.arccos(cosine)
    angle = angle*360/2/np.pi
    
    return angle
def pandas_uaf(array, period):
    array = np.array(array)
    return pd.rolling_mean(array, window=period, min_periods=1, center=True)

def angle_difference_calculate(csv_file):
    df = pd.read_csv(csv_file, skipinitialspace=True)
    df = df.loc[df['success']==1]
    df.reset_index(drop=True, inplace=True)
    # first step is to using unweighted average filter
    aver_period = 5 # five samples as average window
    gaze_0_x = pandas_uaf(df['gaze_0_x'], aver_period)
    gaze_0_y = pandas_uaf(df['gaze_0_y'], aver_period)
    gaze_0_z = pandas_uaf(df['gaze_0_z'], aver_period)
    gaze_1_x = pandas_uaf(df['gaze_1_x'], aver_period)
    gaze_1_y = pandas_uaf(df['gaze_1_y'], aver_period)
    gaze_1_z = pandas_uaf(df['gaze_1_z'], aver_period)
    
    # also, calculate dialtion
    eyeball_lmks_0_x = df[['eye_lmk_x_'+str(i) for i in range(8)]]
    eyeball_lmks_0_x = np.transpose(np.asarray([pandas_uaf(eyeball_lmks_0_x[key], aver_period) for key in eyeball_lmks_0_x.keys()]))
    eyeball_lmks_0_y = df[['eye_lmk_y_'+str(i) for i in range(8)]]
    eyeball_lmks_0_y = np.transpose(np.asarray([pandas_uaf(eyeball_lmks_0_y[key], aver_period) for key in eyeball_lmks_0_y.keys()]))
    pupil_lmks_0_x = df[['eye_lmk_x_'+str(i) for i in range(20,28)]]
    pupil_lmks_0_x = np.transpose(np.asarray([pandas_uaf(pupil_lmks_0_x[key], aver_period) for key in pupil_lmks_0_x.keys()]))
    pupil_lmks_0_y = df[['eye_lmk_y_'+str(i) for i in range(20,28)]]
    pupil_lmks_0_y = np.transpose(np.asarray([pandas_uaf(pupil_lmks_0_y[key], aver_period) for key in pupil_lmks_0_y.keys()]))
    eyeball_lmks_1_x = df[['eye_lmk_x_'+str(i) for i in range(28, 36)]]
    eyeball_lmks_1_x = np.transpose(np.asarray([pandas_uaf(eyeball_lmks_1_x[key], aver_period) for key in eyeball_lmks_1_x.keys()]))
    eyeball_lmks_1_y = df[['eye_lmk_y_'+str(i) for i in range(28, 36)]]
    eyeball_lmks_1_y = np.transpose(np.asarray([pandas_uaf(eyeball_lmks_1_y[key], aver_period) for key in eyeball_lmks_1_y.keys()]))
    pupil_lmks_1_x = df[['eye_lmk_x_'+str(i) for i in range(48, 56)]]
    pupil_lmks_1_x = np.transpose(np.asarray([pandas_uaf(pupil_lmks_1_x[key], aver_period) for key in pupil_lmks_1_x.keys()]))
    pupil_lmks_1_y = df[['eye_lmk_y_'+str(i) for i in range(48,56)]]
    pupil_lmks_1_y = np.transpose(np.asarray([pandas_uaf(pupil_lmks_1_y[key], aver_period) for key in pupil_lmks_1_y.keys()]))
    #angle velocity is calculated for each sample
    # dilation ratio is calculated for each sample
    length = gaze_0_x.shape[0]
    angle_velocities = []
    dialation = []
    timestamps = []
    for i in range(length-1):
        pre = i
        next_one = i+1
        #left eye
        x1_0,y1_0,z1_0 = gaze_0_x[pre], gaze_0_y[pre], gaze_0_z[pre]
        #right eye
        x1_1,y1_1,z1_1 = gaze_1_x[pre], gaze_1_y[pre], gaze_1_z[pre]

        #left eye
        x2_0,y2_0,z2_0 = gaze_0_x[next_one], gaze_0_y[next_one], gaze_0_z[next_one]
        #right eye
        x2_1,y2_1,z2_1 = gaze_1_x[next_one], gaze_1_y[next_one], gaze_1_z[next_one]
        
        L_angle_diff = angle_difference(np.array((x1_0, y1_0, z1_0)), np.array((x2_0, y2_0, z2_0)))
        R_angle_diff = angle_difference(np.array((x1_1, y1_1, z1_1)), np.array((x2_1, y2_1, z2_1)))
        
        average_angle_diff = (L_angle_diff+R_angle_diff)/2
        angle_velocities.append(average_angle_diff*30) #sample rate =30Hz
        #dailation
        left_eye  = calculate_dialation([eyeball_lmks_0_x[i,:], eyeball_lmks_0_y[i,:]], [pupil_lmks_0_x[i,:], pupil_lmks_0_y[i,:]])
        right_eye = calculate_dialation([eyeball_lmks_1_x[i,:], eyeball_lmks_1_y[i,:]], [pupil_lmks_1_x[i,:], pupil_lmks_1_y[i,:]])
        dialation.append((left_eye+right_eye)/2)
        timestamps.append(df.loc[i,'timestamp'])
    return angle_velocities, dialation, timestamps
def calculate_dialation(eyeball, pupil):
    eyeball_x, eyeball_y = eyeball
    pupil_x, pupil_y = pupil
    # origin
    origin_x, origin_y = np.mean(eyeball_x), np.mean(eyeball_y)
    distances = []
    for x,y in zip(eyeball_x, eyeball_y):
        distance = np.sqrt((x-origin_x)**2+(y-origin_y)**2)
        distances.append(distance)
    radius = np.mean(distances)
    area_eyeball = np.pi*(radius**2)
    
    origin_x, origin_y = np.mean(pupil_x), np.mean(pupil_y)
    distances = []
    for x,y in zip(pupil_x, pupil_y):
        distance = np.sqrt((x-origin_x)**2+(y-origin_y)**2)
        distances.append(distance)
    radius = np.mean(distances)
    area_pupil = np.pi*(radius**2)
    
    if area_pupil/area_eyeball >1:
        return 1
    else:
        return area_pupil/area_eyeball
def fixation_classification(angle_velocities):
    #the velocity threshold is 30 degree/s by default
    # the maximum time between fixations is 75ms, which is two samples distance
    threshold = 30.0
    fixations = []
    # simple classification
    for i in range(len(angle_velocities)):
        if angle_velocities[i]>threshold:
            fixations.append(0)
        else:
            fixations.append(1)
    # merge close fixations(less than or equal to two samples); and discard short fixation
    fixation_flag = True if fixations[0]==1 else False
    false_counter=0
    true_counter=0
    for i in range(len(fixations)):
        if fixations[i]==1 and fixation_flag==False:
            fixation_flag=True
            if false_counter<=2:
                fixations[i-false_counter:i]=np.ones(false_counter, dtype=np.int32)
            false_counter=0
            true_counter+=1
        elif fixations[i]==1 and fixation_flag==True:
            true_counter+=1
        elif fixations[i]==0 and fixation_flag==True:
            fixation_flag=False
            if true_counter<=2:
                fixations[i-true_counter:i]=np.zeros(true_counter, dtype=np.int32)
            true_counter=0
            false_counter+=1
        elif fixations[i]==0 and fixation_flag==False:
            false_counter+=1
    return fixations
        
openface_dir = '/newdisk/AVEC2018/downloaded_dataset/openface_faces'
des_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/gaze_angle'
if not path.exists(des_dir):
    os.makedirs(des_dir)
videos =  os.listdir(openface_dir)
for video_name in tqdm(videos):
    pass
    if path.isdir(path.join(openface_dir, video_name)):
        csv_file_path = path.join(openface_dir, video_name, video_name+'.csv')
        angle_velocities, dialations, timestamps = angle_difference_calculate(csv_file_path)
        fixations = fixation_classification(angle_velocities)
        des = path.join(des_dir,video_name+'.csv')
        data = {'frameTime':timestamps, 'angle_velocity':angle_velocities, 'dialation': dialations, 'fixation': fixations}
        df = pd.DataFrame(data=data)
        df = df[['frameTime','angle_velocity','dialation','fixation']]
        df.to_csv(des, index=False)
