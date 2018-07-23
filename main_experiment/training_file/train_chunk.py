#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:30:53 2018

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
from sklearn.model_selection import ParameterGrid
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
from sklearn.metrics import confusion_matrix, recall_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools
feat_type = ['head','pose','gaze_angle','FAUs','DeepSpectrum']
time_interval = [5, 10, 20, 30, 50, 70, 90, 120, 150, 200]
upsample_limit = 300
feat_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features'
metadata_path = '/newdisk/AVEC2018/downloaded_dataset/labels_metadata.csv'
np.random.seed(103)
class Dataset:
    def __init__(self, time_interval, feat_type):
        self.time_interval = time_interval
        self.feat_type = feat_type
        self.wk_dir = os.path.join(feat_dir, feat_type)
        if self.feat_type=='DeepSpectrum':
            sample_rate = 1
        else:
            sample_rate = 30 # 30 Hz
        self.frame_num = time_interval * sample_rate
    def load_my_data(self, gender_type):
        metadata = pd.read_csv(metadata_path, skipinitialspace=True)
        samples = metadata.loc[metadata['Gender']==gender_type]
        print("Loading gender data: "+gender_type)
        train_samples = samples.loc[samples['Partition']=='train']
        dev_samples = samples.loc[samples['Partition']=='dev']
        print("Before upsampling, train set samples: {}, dev set samples: {}".format(len(train_samples), len(dev_samples)))
        # upsample for classes
        train_data, train_target, train_index = self.upsample(train_samples) 
        dev_data, dev_target, dev_index = self.upsample(dev_samples) 
        train_set = {'data': train_data,'target': train_target, 'index': train_index}
        dev_set = {'data': dev_data, 'target': dev_target, 'index': dev_index}
        print("After upsampling, train set shape: {}, dev set shape: {}".format(train_data.shape, dev_data.shape))
        return {'train': train_set, 'dev': dev_set}
    def upsample(self, samples):
        data = []
        target = []
        index = []
        for i in range(1,4):
            class_samples = samples.loc[samples['ManiaLevel']==i]
            #load dataframe once
            dataframes = {}
            for n, row in class_samples.iterrows():
                instance_name = row['Instance_name']
                instance_path = os.path.join(self.wk_dir, instance_name+'.csv')
                dataframe = pd.read_csv(instance_path, skipinitialspace=True, sep='\s+|;|,')
                dataframes[instance_name] =dataframe
            length = len(class_samples)
            idxes = np.random.randint(length, size=int(upsample_limit/3))
            class_samples.reset_index(drop=True, inplace=True)
            for idx in idxes:
                selected_sample = class_samples.loc[idx, :]
                instance_name = selected_sample['Instance_name']
                dataframe = dataframes[instance_name]
                features = dataframe.loc[:,dataframe.keys()[1:]].fillna(0).values
                frame_idxes = sorted(np.random.randint(features.shape[0], size=int(self.frame_num)))
                selected_features = features[frame_idxes, :]
                mean = np.mean(selected_features, axis=0)
                stddev = np.std(selected_features, axis=0)
                kurt = kurtosis(selected_features, axis=0)
                skewness =skew(selected_features, axis=0)
                quartile1 = np.percentile(selected_features, 25, axis=0)
                quartile2 = np.percentile(selected_features, 50, axis=0)
                quartile3 = np.percentile(selected_features, 75, axis=0)   
                data.append(np.concatenate((mean, stddev, kurt, skewness, quartile1, quartile2, quartile3), axis=0))
                target.append(i)
                index.append(instance_name)
        return np.asarray(data), np.asarray(target), index
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def benchmark(clf, X_train, y_train, X_dev, y_dev, dev_index):
    print('_'*80)
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    t0 = time()
    pred = clf.predict(X_dev)
    test_time = time() - t0
    print("evaluate time:  %0.3fs" % test_time)
    
    #get prediction on session level
    dev_videos = np.unique(dev_index)
    y_true = []
    y_pred = []
    for vid in dev_videos:
        idxes = [index for index, video in enumerate(dev_index) if video==vid]
        y_true.append( y_dev[idxes[0]])
        y_pred.append(mode(pred[idxes])[0][0])
        
    score = metrics.recall_score(y_true,y_pred, average='micro')
    print("average recall:   %0.3f" % score)
    
    class_names = ['remission','hypomania','mania']
    print('classification report:')
    print(metrics.classification_report(y_true, y_pred, target_names=class_names))
    return "average recall:   %0.3f" % score

def search_for_best_model(gender, feature, time_int):
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
    results=[]
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")):
#            (KNeighborsClassifier(n_neighbors=10), "kNN"),
#            (RandomForestClassifier(n_estimators=100), "Random forest")):
        print('=' * 80)
        print(name)
        results.append(name)
        results.append(benchmark(clf,X_train, y_train, X_dev, y_dev, dev_index))
    
    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append("%s penalty LinearSVC" % penalty.upper())
        results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                           tol=1e-3), X_train, y_train, X_dev, y_dev, dev_index))
    
        # Train SGD model
        results.append("%s penalty SGDClassifier" % penalty.upper())
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty), X_train, y_train, X_dev, y_dev, dev_index))
    
    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append("Elastic-Net penalty SGDClassifier")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty="elasticnet"), X_train, y_train, X_dev, y_dev, dev_index))
    
    # Train NearestCentroid without threshold
    print('=' * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append('NearestCentroid')
    results.append(benchmark(NearestCentroid(), X_train, y_train, X_dev, y_dev, dev_index))
    
    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01), X_train, y_train, X_dev, y_dev, dev_index))
#    results.append(benchmark(BernoulliNB(alpha=.01), X_train, y_train, X_dev, y_dev, dev_index))
    
    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append("LinearSVC with L1-based feature selection")
    results.append(benchmark(Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                      tol=1e-3))),
      ('classification', LinearSVC(penalty="l2"))]), X_train, y_train, X_dev, y_dev, dev_index))
    # Train votingclassifier
    print('=' * 80)
    print("voting classifier")
    results.append("voting classifier")
    results.append(benchmark(VotingClassifier(estimators=[('ridge', RidgeClassifier(tol=1e-2, solver="lsqr") ),
                                                          ('nb', MultinomialNB(alpha=.01)),
                                                          ('nearest', NearestCentroid())
            ], voting='hard'), X_train, y_train, X_dev, y_dev, dev_index))
    return results
def search_best_hyperparameter():
    file = open('results.txt','w')
    for feat in feat_type:
        for time_int in time_interval:
            file.write("feature type {}; time interval: {}\n".format(feat, time_int))
            file.write("Female\n"+"#" * 80+'\n')
            result = search_for_best_model('F', feat, time_int)
            for line in result:
                file.write(line+'\n')
            file.write("Male\n"+"#" * 80+'\n')
            result = search_for_best_model('M', feat, time_int)
            for line in result:
                file.write(line+'\n')
            file.write("All gender\n"+"#" * 80+'\n')
            result = search_for_best_model('A', feat, time_int)
            for line in result:
                file.write(line+'\n')
        print("Done for {}, {}".format(feat, time_int))
    file.close()     
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
    
    #training using gridsearch
    ticks = time()
    estimator = SVC()
    
    lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_dev = model.transform(X_dev)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))
    # classifications
    param_grid = [
  {'C': [ 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
         0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 10, 100, 1000, 10000, 1e5], 'kernel': ['linear']}
]
    score = 'recall'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    best_score = 0
    for g in ParameterGrid(param_grid):
        estimator.set_params(**g)
        estimator.fit(X_train,y_train) 
        y_true, y_pred = y_dev, estimator.predict(X_dev)
        score = recall_score(y_true, y_pred, average='micro')
        if score>best_score:
            best_score=score
            best_param = g
    print('Time Elapse: {}'.format(time()-ticks))
    
    print("Best parameters set found on development set:")
    print()
    print(best_param)
    print()
    print("Best scores found on development set:")
    print()
    print(best_score)

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    estimator = SVC()
    estimator.set_params(**best_param)
    estimator.fit(X_train, y_train)
    pred =  estimator.predict(X_dev)
    #get prediction on session level
    dev_videos = np.unique(dev_index)
    y_true = []
    y_pred = []
    for vid in dev_videos:
        idxes = [index for index, video in enumerate(dev_index) if video==vid]
        y_true.append(y_dev[idxes[0]])
        y_pred.append(mode(pred[idxes])[0][0])
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()

    # confusion matrix

    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_true, y_pred), classes=class_names,
                          title = 'Confusion matrix without normalization')
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_true, y_pred), classes=class_names, normalize=True,
                          title = 'Confusion matrix')  

pdb.set_trace()
cv_on_SVC('F', 'gaze_angle', 20)