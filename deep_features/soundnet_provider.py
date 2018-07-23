#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:02:36 2018
soundnet feature provider using a dictionary
@author: ddeng
"""

import os.path as path
import glob
import h5py
import numpy as np
conv5_dir = '/newdisk/AVEC2018/downloaded_dataset/soundnet_features/conv5_layer21'
conv7_dir = '/newdisk/AVEC2018/downloaded_dataset/soundnet_features/conv7_layer24'
maxpool5_dir = '/newdisk/AVEC2018/downloaded_dataset/soundnet_features/pool5_layer18'
def read_h5(file_name):
    f = h5py.File(file_name, 'r')
    key = list(f.keys())[0]
    data = np.array(f[key])
    return data    
def soundnet_provider(layer_name):
    if layer_name=='conv5':
        working_dir = conv5_dir
    elif layer_name=='conv7':
        working_dir = conv7_dir
    elif layer_name=='maxpool5':
        working_dir = maxpool5_dir
    soundnet_features = {}
    #scan all video features
    features = glob.glob(path.join(working_dir,'*h5'))
    for feat_file in features:
        video_name = feat_file.split('/')[-1].split('.')[0]
        feat = read_h5(feat_file)
        soundnet_features[video_name] = feat
    return soundnet_features
    

