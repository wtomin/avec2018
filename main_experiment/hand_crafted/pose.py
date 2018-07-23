#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:31:43 2018
pose
@author: ddeng
"""
import numpy as np
import pandas as pd
import os
import os.path as path
from tqdm import tqdm
import glob
def get_needed_joints(csv_file):
    df = pd.read_csv(csv_file, skipinitialspace=True)

    length = len(df)
    keys= ['0_x','0_y','1_x','1_y','2_x','2_y','3_x','3_y','4_x','4_y','5_x','5_y','6_x','6_y','7_x','7_y',
           '14_x','14_y','15_x','15_y','16_x','16_y','17_x','17_y']
    joints = df.loc[:,'frameTime']
    for key in keys:
        try:
            col = df[key]
        except:
            col = pd.Series(0, index = np.arange(length))
        joints = pd.concat([joints, col], axis=1)
    joints.reset_index(drop=True, inplace=True)
    return joints
pose_dir = '/newdisk/AVEC2018/downloaded_dataset/recordings/video_pose'
des_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/pose'
if not path.exists(des_dir):
    os.makedirs(des_dir)

pose_paths = glob.glob(os.path.join(pose_dir,'*.csv'))

for pose_path in tqdm(pose_paths):
    pass
    video_name = pose_path.split('/')[-1].split('.')[0]
    joints_csv = get_needed_joints(pose_path)
    des = os.path.join(des_dir, video_name+'.csv')
    joints_csv.to_csv(des, index=False)