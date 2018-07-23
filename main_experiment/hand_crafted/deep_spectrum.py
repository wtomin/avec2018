#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:00:12 2018
deep spectrum
@author: ddeng
"""
import os
import pdb
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
n_cluster = 10
random_state=220
in_dir = '/newdisk/AVEC2018/downloaded_dataset/baseline_features/LLDs_audio_DeepSpectrum_turns'
outdir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/deep_spectrum'
if not os.path.exists(outdir):
    os.makedirs(outdir)
def get_clusters(sample):
    features = sample   
    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state, n_jobs=-1)
    kmeans.fit(features)
    return kmeans.cluster_centers_

for video_file in tqdm(glob.glob(os.path.join(in_dir, '*.csv'))):
    pass
    video_name =video_file.split('/')[-1].split('.')[0]
    df = pd.read_csv(video_file, sep=';', skipinitialspace=True)
    df = df.loc[:, df.keys()[1:]]
    features = df.values
    clusters = get_clusters(features)
    des = os.path.join(outdir, video_name+'.npy')
    np.save(des, clusters)
    