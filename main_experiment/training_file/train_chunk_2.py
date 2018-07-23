#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:50:52 2018

@author: ddeng
"""

import pandas as pd
import pdb
import os
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from time import time
from scipy.stats import skew, kurtosis, mode
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from my_utils import plot_confusion_matrix, Dataset,cv_on_SVC_feat_selection
feat_type = ['head','pose','gaze_angle','FAUs']
time_interval = [5, 10, 20, 30, 50, 70, 90, 120, 150, 200]
def cv_on_SVC(gender, feature, time_int):
    dataset = Dataset(feat_type=feature, time_interval=time_int)
    female_data = dataset.load_my_data('F')
    male_data = dataset.load_my_data('M')
    if gender=='F':
        X_train, y_train = female_data['train']['data'], female_data['train']['target']
        X_dev, y_dev, dev_index = female_data['dev']['data'], female_data['dev']['target'], female_data['dev']['index'] 
    elif gender=='M':
        X_train, y_train = male_data['train']['data'], male_data['train']['target']
        X_dev, y_dev, dev_index = male_data['dev']['data'], male_data['dev']['target'], male_data['dev']['index'] 
    elif gender=='A':
        X_train = np.concatenate((female_data['train']['data'], male_data['train']['data']), axis=0)
        y_train = np.concatenate((female_data['train']['target'], male_data['train']['target']), axis=0)
        X_dev = np.concatenate((female_data['dev']['data'], male_data['dev']['data']), axis=0)
        y_dev = np.concatenate((female_data['dev']['target'], male_data['dev']['target']), axis=0)
        dev_index= female_data['dev']['index']
        dev_index.extend(male_data['dev']['index'])
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_dev = scaler.transform(X_dev)
    
    y_pred_probs = cv_on_SVC_feat_selection(X_train, y_train, X_dev, y_dev, dev_index)
    
