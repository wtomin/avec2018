#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:11:42 2018
AUs 
@author: ddeng
"""
import numpy as np
import pandas as pd
import os
import os.path as path
from tqdm import tqdm
def get_FAUs(csv_file):
    df = pd.read_csv(csv_file, skipinitialspace=True)
    df = df.loc[df['success']==1]
    df.reset_index(drop=True, inplace=True)
    keys = ['timestamp','AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r','AU07_r', 
           'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
           'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r',
           'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 
           'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
    FAUs = df.loc[:, keys]
    return FAUs
openface_dir = '/newdisk/AVEC2018/downloaded_dataset/openface_faces'
des_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/FAUs'
if not path.exists(des_dir):
    os.makedirs(des_dir)
videos =  os.listdir(openface_dir)
for video_name in tqdm(videos):
    pass
    if path.isdir(path.join(openface_dir, video_name)):
        csv_file_path = path.join(openface_dir, video_name, video_name+'.csv')
        FAUs = get_FAUs(csv_file_path)
        des = path.join(des_dir,video_name+'.csv')
        FAUs.to_csv(des, index=False)