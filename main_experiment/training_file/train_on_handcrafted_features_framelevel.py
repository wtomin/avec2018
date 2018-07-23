#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 22:15:40 2018
train_on_handcrafted features on frame level
@author: ddeng
"""
import pandas as pd
import numpy as np
import os
import glob
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
import time
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from scipy.stats import mode
# head, pose, gaze, au, deep_face, soundnet, deepspectrum, eGemaps
metadata_path = '/newdisk/AVEC2018/downloaded_dataset/labels_metadata.csv'
head_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/head'
pose_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/pose'
FAUs_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/FAUs'
gaze_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/gaze_angle'
deepspectrum_dir = '/newdisk/AVEC2018/downloaded_dataset/baseline_features/LLDs_audio_DeepSpectrum_turns'
eGeMAPS_dir = '/newdisk/AVEC2018/downloaded_dataset/baseline_features/features_audio_eGeMAPS_turns'

def load_frame_level_dataset(featdir):
    metadata = pd.read_csv(metadata_path, skipinitialspace=True)
    Female_samples = metadata.loc[metadata['Gender']=='F']
    Male_samples = metadata.loc[metadata['Gender']=='M']
    # female data
    female_data, female_target, female_vid_index = get_dataset(Female_samples, featdir)
    male_data, male_target, male_vid_index = get_dataset(Male_samples, featdir)
    return (female_data, female_target, female_vid_index) ,(male_data, male_target, male_vid_index)
def upsample(data, target, vid_index):
    # upsample according to the mania level
    train_data, dev_data = data
    train_target, dev_target = target
    train_vid_index, dev_vid_index = vid_index
    
    train_target_m = train_target['mania']
    class_ratio_1 = np.sum([train_target_m[train_target_m==1]])
    class_ratio_2 = np.sum([train_target_m[train_target_m==2]])/2
    class_ratio_3 = np.sum([train_target_m[train_target_m==3]])/3

    return None
def get_dataset(Female_samples, featdir):
    train_data_total = []
    train_target_m_total = []
    train_target_y_total = []
    train_vid_index = []
    dev_data_total = []
    dev_target_m_total = []
    dev_target_y_total = []
    dev_vid_index = []
    for index, row in Female_samples.iterrows():
        instance_name = row['Instance_name']
        is_train = row['Partition']=='train'
        target_m = row['ManiaLevel']
        target_y = row['Total_YMRS']
        featpath = os.path.join(featdir, instance_name+'.csv')
        df = pd.read_csv(featpath, skipinitialspace=True)
        feat_list = ['magnitude','derivative','pitch','yaw','roll']
        if is_train:
            train_data = df.loc[:, feat_list].dropna().values
            train_target_m = np.ones((train_data.shape[0],), dtype=np.int)*target_m
            train_target_y = np.ones((train_data.shape[0],), dtype=np.int)*target_y
            train_data_total.append(train_data)
            train_target_m_total.append(train_target_m)
            train_target_y_total.append(train_target_y)
            train_vid_index.extend([instance_name for _ in range(train_data.shape[0])])
        else:
            dev_data = df.loc[:, feat_list].dropna().values
            dev_target_m = np.ones((dev_data.shape[0],), dtype=np.int)*target_m
            dev_target_y = np.ones((dev_data.shape[0],), dtype=np.int)*target_y
            dev_data_total.append(dev_data)
            dev_target_m_total.append(dev_target_m)
            dev_target_y_total.append(dev_target_y)
            dev_vid_index.extend([instance_name for _ in range(dev_data.shape[0])])
    train_data = np.concatenate([arr for arr in train_data_total], axis =0)
    train_target_m = np.concatenate([arr for arr in train_target_m_total], axis=0)
    train_target_y = np.concatenate([arr for arr in train_target_y_total], axis=0)
    dev_data = np.concatenate([arr for arr in dev_data_total], axis =0)
    dev_target_m = np.concatenate([arr for arr in dev_target_m_total], axis=0)
    dev_target_y = np.concatenate([arr for arr in dev_target_y_total], axis=0)
    female_data = [train_data, dev_data]
    female_target = [{'mania':train_target_m, 'ymrs': train_target_y}, {'mania':dev_target_m, 'ymrs': dev_target_y}]
    female_vid_index= [train_vid_index, dev_vid_index]
    return female_data, female_target, female_vid_index
def train_on_head(feature_name):
    featdir = head_dir
    print('Loading head data...')
    import pdb
    pdb.set_trace()
    female_dataset, male_dataset = load_frame_level_dataset(featdir)
    female_data, female_target, female_vid_index = female_dataset
    X_train, X_dev = female_data
    y_train, y_dev = female_target[0]['mania'], female_target[1]['mania']
    X_train = preprocessing.scale(X_train)
    X_dev = preprocessing.scale(X_dev)
    estimator = BernoulliNB()
    timestamp = time.time()
    estimator.fit(X_train,y_train)
    print("Time elapse: {}".format(time.time()-timestamp))
    y_true, y_pred = y_dev, estimator.predict(X_dev)
    y_true, y_pred = get_major_vote_class(y_true, y_pred, female_vid_index[1])
    class_names = ['remission','hypomania','mania']
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()
def get_major_vote_class(y_true, y_pred, vid_index):
    merged_y_true = []
    merged_y_pred = []
    unique_vids = np.unique(vid_index)
    for vid in unique_vids:
        idx = [index for index, name in enumerate(vid_index) if name==vid]
        merged_y_true.append(y_true[idx[0]])
        selected_y_pred = y_pred[idx]
        merged_y_pred.append(mode(selected_y_pred)[0][0])
    return merged_y_true, merged_y_pred
train_on_head('head')