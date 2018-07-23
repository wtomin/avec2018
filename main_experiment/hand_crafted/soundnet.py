#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:52:40 2018

@author: ddeng
"""
import os
import sys
import numpy as np
pathA = '/newdisk/AVEC2018/downloaded_dataset/deep_features'
sys.path.append(os.path.abspath(pathA))
from soundnet_provider import soundnet_provider
from sklearn.cluster import KMeans
from tqdm import tqdm
import pdb
pdb.set_trace()
soundnet_layer = 'maxpool5'
n_cluster = 20
random_state=220

def get_clusters(sample):
    features = np.transpose(sample)    
    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state, n_jobs=-1)
    kmeans.fit(features)
    return kmeans.cluster_centers_
#min length=14, max length=995

#outdir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/soundnet_conv5'
#sound_features = soundnet_provider(soundnet_layer)
#if not os.path.exists(outdir):
#    os.makedirs(outdir)
#for sample_name in tqdm(sound_features.keys()):
#    sample = sound_features[sample_name]
#    clusters = get_clusters(sample)
#    des = os.path.join(outdir, sample_name+'.npy')
#    np.save(des, clusters)
#outdir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/soundnet_conv7'
#sound_features = soundnet_provider(soundnet_layer)
#if not os.path.exists(outdir):
#    os.makedirs(outdir)
#lengths = []
#for sample_name in tqdm(sound_features.keys()):
#    #min length=8, max length=498
#    sample = sound_features[sample_name]
#    clusters = get_clusters(sample)
#    des = os.path.join(outdir, sample_name+'.npy')
#    np.save(des, clusters)
outdir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/soundnet_pool5'
sound_features = soundnet_provider(soundnet_layer)
if not os.path.exists(outdir):
    os.makedirs(outdir)
lengths = []
for sample_name in tqdm(sound_features.keys()):
    #min length=27, max length=1988
    sample = sound_features[sample_name]
    lengths.append(sample.shape[1])
    clusters = get_clusters(sample)
    des = os.path.join(outdir, sample_name+'.npy')
    np.save(des, clusters)
