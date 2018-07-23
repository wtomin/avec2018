#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:18:21 2018
openface
@author: ddeng
"""
import os
import sys

pathA = '/newdisk/AVEC2018/downloaded_dataset/deep_features'
sys.path.append(os.path.abspath(pathA))
from openface_provider import openface_provider
import pdb
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import random

n_cluster = 20
random_state=220

def get_clusters(sample):
    feature_list = []
    for feat_path in sample:
        feature_list.append(np.load(feat_path))
    features = np.asarray(feature_list)

    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state, n_jobs=-1)
    kmeans.fit(features)
    return kmeans.cluster_centers_
def get_samples(sample):
    feature_list = []
    for feat_path in sample:
        feature_list.append(np.ndarray.flatten(np.load(feat_path)))
    features = np.asarray(feature_list)

    return features
#dictionary = openface_provider('vgg16','fc6', False)
#outdir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/vgg16_fc6'
#if not os.path.exists(outdir):
#    os.makedirs(outdir)
#for sample_name in tqdm(dictionary.keys()):
#    sample = dictionary[sample_name]
#    clusters = get_clusters(sample)
#    des = os.path.join(outdir, sample_name+'.npy')
#    np.save(des, clusters)
    
#dictionary = openface_provider('vgg16','fc7', False)
#
#outdir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/vgg16_fc7'
#if not os.path.exists(outdir):
#    os.makedirs(outdir)
#for sample_name in tqdm(dictionary.keys()):
#    sample = dictionary[sample_name]
#    clusters = get_clusters(sample)
#    des = os.path.join(outdir, sample_name+'.npy')
#    np.save(des, clusters)
#
dictionary = openface_provider('vgg16','pool5', False)
outdir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/vgg16_pool5'
if not os.path.exists(outdir):
    os.makedirs(outdir)
for sample_name in tqdm(dictionary.keys()):
    sample = dictionary[sample_name]
    # time is short, only randomly select 20 frames
    r_frames = sorted(random.sample(sample, n_cluster))
    features = get_samples(r_frames)
    des = os.path.join(outdir, sample_name+'.npy')
    np.save(des, features)