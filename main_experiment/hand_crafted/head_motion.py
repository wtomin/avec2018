#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:36:44 2018
head position and derivatives
@author: ddeng
"""
import os.path as path
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
def head_motion_calculate(csv_file):
    df = pd.read_csv(csv_file, skipinitialspace=True)
    dataframe = df.loc[df['success']==1]
    dataframe.reset_index(drop=True, inplace=True)
    x,y,z = dataframe['pose_Tx'], dataframe['pose_Ty'], dataframe['pose_Tz']
    aver_period = 5
    signal = [pandas_uaf(x,aver_period) ,pandas_uaf(y, aver_period), pandas_uaf(z, aver_period)]
    derivatives = calculate_derivatives(signal)
    magnitudes = calculate_magnitude(signal)
    pitch = pandas_uaf(dataframe['pose_Rx'], aver_period)
    yaw = pandas_uaf(dataframe['pose_Ry'], aver_period)
    roll = pandas_uaf(dataframe['pose_Rz'], aver_period)
    length = derivatives.shape[0] if derivatives.shape[0]<magnitudes.shape[0] else magnitudes.shape[0]
    return magnitudes[:length], derivatives[:length], dataframe.loc[:length-1, 'timestamp'].values, pitch[:length], yaw[:length], roll[:length]
def calculate_derivatives(signal):
    # the input is :(x,y,z)
    derivatives = []
    for i in range(len(signal[0])):
        t_minus_1 = i
        t = (i+1)
        if t>=len(signal[0]):
            break
        x1,y1,z1 = signal[0][t_minus_1], signal[1][t_minus_1], signal[2][t_minus_1]
        x2,y2,z2 = signal[0][t], signal[1][t], signal[2][t]
        diff = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        derivatives.append(diff)
    return np.asarray(derivatives)

def calculate_magnitude( signal):
    #input: (x,y,z)
    magnitude = []
    for i in range(len(signal[0])):
        mag = np.sqrt(signal[0][i]**2+ signal[1][i]**2+ signal[2][i]**2)
        magnitude.append(mag)
        
    return np.asarray(magnitude)
def pandas_uaf(array, period):
    array = np.array(array)
    return pd.rolling_mean(array, window=period, min_periods=1, center=True)

openface_dir = '/newdisk/AVEC2018/downloaded_dataset/openface_faces'
des_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/head'
if not path.exists(des_dir):
    os.makedirs(des_dir)
videos =  os.listdir(openface_dir)
for video_name in tqdm(videos):
    pass
    if path.isdir(path.join(openface_dir, video_name)):
        csv_file_path = path.join(openface_dir, video_name, video_name+'.csv')
        magnitudes, derivatives, timestamps, pitch, yaw, roll = head_motion_calculate(csv_file_path)
        des = path.join(des_dir,video_name+'.csv')
        data = {'frameTime': timestamps, 'magnitude': magnitudes, 'derivative':derivatives, 'pitch':pitch, 'yaw': yaw, 'roll': roll}
        df = pd.DataFrame(data=data)
        df = df[['frameTime','magnitude', 'derivative','pitch', 'yaw', 'roll']]
        df.to_csv(des, index=False)