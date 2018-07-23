#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:11:05 2018

@author: ddeng
"""

import pandas as pd
import pdb
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Perceptron
import itertools
import subprocess
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestCentroid
import os
import copy
import seaborn as sns
np.random.seed(220) #do not change seed
FEATURE_LIST = ['head','pose', 'gaze_angle', 'face', 'FAUs', 'egmaps','mfcc','speech']
wk_dir ='/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features'
metadata_path = '/newdisk/AVEC2018/downloaded_dataset/labels_metadata.csv'
test_meta_path = '/newdisk/AVEC2018/downloaded_dataset/test_metadat.csv'
def load_my_data():
    df = pd.read_csv('hand_crafted_features_va.csv') 
    n_sample = len(df['Instance_name'])
    n_feature = len(df.keys()[7:])
    data = np.empty((n_sample, n_feature) , dtype = np.float64)
    target = np.empty((n_sample, ), dtype = np.int)
    
    for index, row in df.iterrows():
        data[index] = np.asarray(df.loc[index, df.keys()[7:]], dtype =np.float64)
        target[index] = np.asarray(df.loc[index, 'ManiaLevel'])
    full_dict = {'data': data, 'target': target}
    train_df = df.loc[df['Partition']=='train']
    train_df.reset_index(drop =True, inplace=True)
    train_n_sample = len(train_df)
    dev_df = df.loc[df['Partition']=='dev']
    dev_df.reset_index(drop =True, inplace=True)
    dev_n_sample = len(dev_df)
    
    data = np.empty((train_n_sample, n_feature) , dtype = np.float64)
    target = np.empty((train_n_sample, ), dtype = np.int)
    target_y =  np.empty((train_n_sample, ), dtype = np.int)
    for index, row in train_df.iterrows():
        data[index] = np.asarray(train_df.loc[index, train_df.keys()[7:]], dtype =np.float64)
        target[index] = np.asarray(train_df.loc[index, 'ManiaLevel'])
        target_y[index] = np.asarray(train_df.loc[index,'Total_YMRS'])
    train_set = [data, target, target_y]
    data = np.empty((dev_n_sample, n_feature) , dtype = np.float64)
    target = np.empty((dev_n_sample, ), dtype = np.int)
    target_y =  np.empty((dev_n_sample, ), dtype = np.int)
    for index, row in dev_df.iterrows():
        data[index] = np.asarray(dev_df.loc[index, dev_df.keys()[7:]], dtype =np.float64)
        target[index] = np.asarray(dev_df.loc[index, 'ManiaLevel'])
        target_y[index] = np.asarray(dev_df.loc[index,'Total_YMRS'])
    dev_set = [data, target, target_y]
    return full_dict, {'train':train_set, 'dev': dev_set}

def load_dataset(features_list):
    dfs = []
    for feat in features_list:
        csv_file = os.path.join(wk_dir, feat+'.csv')
        df = pd.read_csv(csv_file, skipinitialspace=True)
        dfs.append(df)
    if len(dfs)==1:
        dataframe = dfs[0]
    else:
        dataframe = dfs[0]
        for df in dfs[1:]:
            dataframe = dataframe.merge(df, how='outer')
    n_feature = len(dataframe.keys()[7:])
    
    train_df = dataframe.loc[dataframe['Partition']=='train']
    train_df.reset_index(drop =True, inplace=True)
    dev_df = dataframe.loc[dataframe['Partition']=='dev']
    dev_df.reset_index(drop =True, inplace=True)
    train_n_sample = len(train_df)
    dev_n_sample = len(dev_df)
    
    data = np.empty((train_n_sample, n_feature) , dtype = np.float64)
    target = np.empty((train_n_sample, ), dtype = np.int)
    target_y =  np.empty((train_n_sample, ), dtype = np.int)
    for index, row in train_df.iterrows():
        data[index] = np.asarray(train_df.loc[index, train_df.keys()[7:]], dtype =np.float64)
        target[index] = np.asarray(train_df.loc[index, 'ManiaLevel'])
        target_y[index] = np.asarray(train_df.loc[index,'Total_YMRS'])
    train_set = [data, target, target_y, train_df.keys()[7:]]
    data = np.empty((dev_n_sample, n_feature) , dtype = np.float64)
    target = np.empty((dev_n_sample, ), dtype = np.int)
    target_y =  np.empty((dev_n_sample, ), dtype = np.int)
    for index, row in dev_df.iterrows():
        data[index] = np.asarray(dev_df.loc[index, dev_df.keys()[7:]], dtype =np.float64)
        target[index] = np.asarray(dev_df.loc[index, 'ManiaLevel'])
        target_y[index] = np.asarray(dev_df.loc[index,'Total_YMRS'])
    dev_set = [ data, target, target_y]
    # test set
    dfs = []
    for feat in features_list:
        csv_file = os.path.join(wk_dir, feat+'_test.csv')
        df = pd.read_csv(csv_file, skipinitialspace=True)
        dfs.append(df)
    if len(dfs)==1:
        dataframe = dfs[0]
    else:
        dataframe = dfs[0]
        for df in dfs[1:]:
            dataframe = dataframe.merge(df, how='outer')
    n_feature = len(dataframe.keys()[2:])
    n_sample = len(dataframe)
    data = np.empty((n_sample, n_feature))
    for index, row in dataframe.iterrows():
        data[index] = np.asarray(dataframe.loc[index , dataframe.keys()[2:]], dtype=np.float64)
    test_set = data
    return {'train':train_set, 'dev': dev_set, 'test': test_set}
def load_gender():
    dataframe = pd.read_csv(metadata_path, skipinitialspace=True)
    
    train_df = dataframe.loc[dataframe['Partition']=='train']
    train_df.reset_index(drop =True, inplace=True)
    dev_df = dataframe.loc[dataframe['Partition']=='dev']
    dev_df.reset_index(drop =True, inplace=True)
    train_n_sample = len(train_df)
    dev_n_sample = len(dev_df)
    # M: 1, F: 0
    data = np.empty((train_n_sample, 1) , dtype = np.float64)
    for index, row in train_df.iterrows():
        if train_df.loc[index, 'Gender']=='M':
            data[index] = np.asarray(1)
        else:
            data[index] = np.asarray(0)
    train_set = data
    
    data = np.empty((dev_n_sample,1 ) , dtype = np.float64)
    for index, row in dev_df.iterrows():
        if dev_df.loc[index, 'Gender']=='M':
            data[index] = np.asarray(1)
        else:
            data[index] = np.asarray(0)
    dev_set = data
    
     #test set
    dataframe = pd.read_csv(test_meta_path, skipinitialspace=True)  
    test_n_sample = len(dataframe)
    data = np.empty((test_n_sample,1 ) , dtype = np.float64)
    for index, row in dataframe.iterrows():
        if dataframe.loc[index, 'Gender']=='M':
            data[index] = np.asarray(1)
        else:
            data[index] = np.asarray(0)
    test_set = data
    
    return {'train':train_set, 'dev': dev_set, 'test': test_set}
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
def cv_on_RF(X_train, y_train, X_test, y_test):
    ticks = time.time()
    estimator = RandomForestClassifier(random_state=220)

    # classifications
    param_grid = {
            'n_estimators': [100, 200, 400],
            'max_features': ['auto', 'sqrt', 'log2']}
    score = 'recall'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s_micro' % score)
    clf.fit(X_train, y_train)
    print('Time Elapse: {}'.format(time.time()-ticks))
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Best scores found on development set:")
    print()
    print(clf.best_score_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()
    score_test = recall_score(y_true, y_pred, average='micro')
    result_dict = {'classifier': 'RF', 'best_params': clf.best_params_, 'best_score':clf.best_score_, 'dev_score': score_test}
    return result_dict
def cv_on_SVC(X_train, y_train, X_test, y_test):

    ticks = time.time()
    estimator =SVC(kernel='linear')

    # classifications
    param_grid = [
  {'C': [ 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
         0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 10, 100, 1000, 10000, 1e5], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7], 'kernel': ['rbf']},
]
    score = 'recall'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s_micro' % score)
    clf.fit(X_train, y_train)
    print('Time Elapse: {}'.format(time.time()-ticks))
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Best scores found on development set:")
    print()
    print(clf.best_score_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()
    score_test = recall_score(y_true, y_pred, average='micro')
    result_dict = {'classifier': 'SVC', 'best_params': clf.best_params_, 'best_score':clf.best_score_, 'dev_score': score_test}
    return result_dict

def cv_on_NB(X_train, y_train, X_test, y_test):
    ticks = time.time()
    estimator = MultinomialNB()

    # classifications
    param_grid = [
  {'alpha': [ 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
         0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 10]}
    
]
    score = 'recall'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s_micro' % score)
    clf.fit(X_train, y_train)
    print('Time Elapse: {}'.format(time.time()-ticks))
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Best scores found on development set:")
    print()
    print(clf.best_score_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()
    score_test = recall_score(y_true, y_pred, average='micro')
    result_dict = {'classifier': 'NB', 'best_params': clf.best_params_, 'best_score':clf.best_score_, 'dev_score': score_test}
    return result_dict
def get_class_from_ymrs(ymrs):
    if ymrs<=7:
        return 1
    elif ymrs>=20:
        return 3
    else:
        return 2
def cv_on_ridge_regression():
    _, dataset= load_my_data()
    X_train,_,y_train = dataset['train']
    X_test, _, y_test = dataset['dev']
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    
    ticks = time.time()
    estimator = Ridge()
    lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))
    param_grid = {'alpha': np.arange(640, 670,0.1)}
    score = 'neg_mean_squared_error'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=6, scoring=score)
    clf.fit(X_train, y_train)
    print('Time Elapse: {}'.format(time.time()-ticks))
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Best scores found on development set:")
    print()
    print(clf.best_score_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    pred =  clf.predict(X_test)
    y_true = []
    y_pred = []
    for i in range(len(y_test)):
        y_true.append(get_class_from_ymrs(y_test[i]))
        y_pred.append(get_class_from_ymrs(pred[i]))
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()   
    # confusion matrix

    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_true, y_pred), classes=class_names,
                          title = 'Confusion matrix without normalization')
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_true, y_pred), classes=class_names, normalize=True,
                          title = 'Confusion matrix')  
def cv_on_NC():
    _, dataset= load_my_data()
    X_train,y_train,_ = dataset['train']
    X_test, y_test,_ = dataset['dev']
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    # bianry classification
#    y[y==3]=2
    # feature normalization

    ticks = time.time()
    
    estimator = NearestCentroid()

    lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))

    # classifications
    param_grid = [
  {'shrink_threshold': [ None, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2]}
]
    score = 'recall'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s_micro' % score)
    clf.fit(X_train, y_train)
    print('Time Elapse: {}'.format(time.time()-ticks))
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Best scores found on development set:")
    print()
    print(clf.best_score_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()

    # confusion matrix

    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names,
                          title = 'Confusion matrix without normalization')
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names, normalize=True,
                          title = 'Confusion matrix')  

    line = "Best: %f using %s" % (clf.best_score_, clf.best_params_)
    return [classification_report(y_true, y_pred, target_names=class_names),line]
def cv_on_models():
    results = []
    for i in tqdm(np.arange(1, 10)):
        pass
        feature_iteration = itertools.combinations(FEATURE_LIST,i)
        for feature_list in feature_iteration:
            dataset = load_dataset(feature_list)
            X_train,y_train,_ = dataset['train']
            X_test, y_test,_ = dataset['dev']
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
    
            lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
            model = SelectFromModel(lsvc, prefit=True)
            num_f_bef = X_train.shape
            X_train = model.transform(X_train)
            X_test = model.transform(X_test)
        
            print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))
        
            result_SVC = cv_on_SVC(X_train, y_train, X_test, y_test)
            result_RF = cv_on_RF(X_train, y_train, X_test, y_test)
            result_NB = cv_on_NB(X_train, y_train, X_test, y_test)
            dev_scores = [result_SVC['dev_score'], result_RF['dev_score'], result_NB['dev_score']]
            max_id = np.argmax(dev_scores)
            if max_id==0:
                results.append([feature_list, result_SVC])
            elif max_id ==1 :
                results.append([feature_list, result_RF])
            elif max_id ==2:
                results.append([feature_list, result_NB])
        
    with open('results_201807021500.pkl','wb') as file:
        pickle.dump(results, file)
def sort_results(results):
    dev_scores_list = []
    for result in results:
        cv_dict = result[1]
        dev_score = cv_dict['dev_score']
        dev_scores_list.append(dev_score)
    sort_index = list(np.argsort(dev_scores_list))
    return [results[index] for index in sort_index]
def sort_results_on_cv_score(results):
    cv_scores_list = []
    for result in results:
        cv_dict = result[1]
        cv_score = cv_dict['best_score']
        cv_scores_list.append(cv_score)
    sort_index = list(np.argsort(cv_scores_list))
    return [results[index] for index in sort_index]
def train_post_on_top_two():
    results = pickle.load(open('results_201807021500.pkl','rb'))
    #results = pickle.load(open('results.pkl','rb')) gives 0.62
    #results = pickle.load(open('results_201806301017.pkl','rb')) GIVES 0.55
    #results = pickle.load(open('results_201807021500.pkl','rb')) gives 0.58
    top2_results = sort_results(results=results)[-2:]
    proba_list = []
    for result in top2_results:
        feature_list = result[0]
        cv_dict = result[1]
        classifier = cv_dict['classifier']
        best_params = cv_dict['best_params']
        print(result)
        
        dataset = load_dataset(feature_list)
        X_train,y_train,_,_ = dataset['train']
        X_dev, y_dev,_ = dataset['dev']
        X_test = dataset['test']
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)

        lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
        model = SelectFromModel(lsvc, prefit=True)
        num_f_bef = X_train.shape
        X_train = model.transform(X_train)
        X_dev = model.transform(X_dev)
        X_test = model.transform(X_test)
    
        print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))
        
        if classifier=='NB':
            estimator = MultinomialNB()
        elif classifier=='SVC':
            estimator = SVC(probability=True)
        elif classifier=='RF':
            estimator = RandomForestClassifier(random_state=220)
        
        estimator.set_params(**best_params)
        
        # train
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_dev)
        print("Average recall: {}".format(recall_score(y_dev, y_pred, average='micro')))
        # predict probabilities as features
        X_train_proba = estimator.predict_proba(X_train)
        X_dev_proba = estimator.predict_proba(X_dev)
        X_test_proba = estimator.predict_proba(X_test)
        
        proba_list.append([X_train_proba, X_dev_proba, X_test_proba])
        
    X_train_proba_merge = np.concatenate([x[0] for x in proba_list], axis =1)
    X_dev_proba_merge = np.concatenate([x[1] for x in proba_list], axis =1)
    X_test_proba_merge = np.concatenate([x[2] for x in proba_list], axis =1)
    
    X_train_proba_merge = preprocessing.scale(X_train_proba_merge)
    X_dev_proba_merge = preprocessing.scale(X_dev_proba_merge)
    X_test_proba_merge = preprocessing.scale(X_test_proba_merge)
    estimator = SVC(C=1, kernel='linear')
    class_names = ['remission','hypomania','mania']

    estimator.fit(X_train_proba_merge, y_train)
    y_true, y_pred = y_dev, estimator.predict(X_dev_proba_merge)
    print("1: {}, 2:{}, 3:{}".format(sum(y_pred[y_pred==1]), 
          sum(y_pred[y_pred==2])/2, sum(y_pred[y_pred==3])/3))
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()

    # confusion matrix

    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_dev, y_pred), classes=class_names,
                          title = 'Confusion matrix without normalization')
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_dev, y_pred), classes=class_names, normalize=True,
                          title = 'Confusion matrix') 
    
    #generate test result
    y_pred_test = estimator.predict(X_test_proba_merge)
    print("1: {}, 2:{}, 3:{}".format(sum(y_pred_test[y_pred_test==1]), 
          sum(y_pred_test[y_pred_test==2])/2, sum(y_pred_test[y_pred_test==3])/3))
    test_csv = '/newdisk/AVEC2018/downloaded_dataset/test_metadat.csv'
    test_df = pd.read_csv(test_csv)
    test_df.loc[:,'Predicted_label'] = y_pred_test
    test_df = test_df.loc[:, ['Instance_name', 'Predicted_label']]    
    des = 'BDS_NISL_6.csv'
    test_df.to_csv(des, index=False, sep=',')
def train_post_on_top_two_gender_diff():
    results = pickle.load(open('results_201806301017.pkl','rb'))
    #results = pickle.load(open('results.pkl','rb')) gives 0.62
    #results = pickle.load(open('results_201806301017.pkl','rb')) GIVES 0.55
    #results = pickle.load(open('results_201807021500.pkl','rb')) gives 0.58
    top2_results = sort_results(results=results)[-2:]
    proba_list = []
    for result in top2_results:
        feature_list = result[0]
        cv_dict = result[1]
        classifier = cv_dict['classifier']
        best_params = cv_dict['best_params']
        print(result)
        
        dataset = load_dataset(feature_list)
        X_train,y_train,_,_ = dataset['train']
        X_dev, y_dev,_ = dataset['dev']
        X_test = dataset['test']
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)

        lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
        model = SelectFromModel(lsvc, prefit=True)
        num_f_bef = X_train.shape
        X_train = model.transform(X_train)
        X_dev = model.transform(X_dev)
        X_test = model.transform(X_test)
    
        print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))
        
        if classifier=='NB':
            estimator = MultinomialNB()
        elif classifier=='SVC':
            estimator = SVC(probability=True)
        elif classifier=='RF':
            estimator = RandomForestClassifier(random_state=220)
        
        estimator.set_params(**best_params)
        # add gender difference
        gender_info = load_gender()
        gender_train, gender_dev, gender_test = gender_info['train'], gender_info['dev'], gender_info['test']
        # add gender to features
        X_train = np.concatenate((X_train, gender_train), axis=-1)
        X_dev = np.concatenate((X_dev, gender_dev), axis=-1)
        X_test = np.concatenate((X_test, gender_test), axis=-1)
        # train
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_dev)
        print("Average recall: {}".format(recall_score(y_dev, y_pred, average='micro')))
        # predict probabilities as features
        X_train_proba = estimator.predict_proba(X_train)
        X_dev_proba = estimator.predict_proba(X_dev)
        X_test_proba = estimator.predict_proba(X_test)
        
        proba_list.append([X_train_proba, X_dev_proba, X_test_proba])
        
    X_train_proba_merge = np.concatenate([x[0] for x in proba_list], axis =1)
    X_dev_proba_merge = np.concatenate([x[1] for x in proba_list], axis =1)
    X_test_proba_merge = np.concatenate([x[2] for x in proba_list], axis =1)
    
    X_train_proba_merge = preprocessing.scale(X_train_proba_merge)
    X_dev_proba_merge = preprocessing.scale(X_dev_proba_merge)
    X_test_proba_merge = preprocessing.scale(X_test_proba_merge)
    
    estimator = SVC(C=1, kernel='linear')
    class_names = ['remission','hypomania','mania']

    estimator.fit(X_train_proba_merge, y_train)
    y_true, y_pred = y_dev, estimator.predict(X_dev_proba_merge)

    print("1: {}, 2:{}, 3:{}".format(sum(y_pred[y_pred==1]), 
          sum(y_pred[y_pred==2])/2, sum(y_pred[y_pred==3])/3))
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()

    # confusion matrix

    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_dev, y_pred), classes=class_names,
                          title = 'Confusion matrix without normalization')
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_dev, y_pred), classes=class_names, normalize=True,
                          title = 'Confusion matrix') 
    
    #generate test result
    y_pred_test = estimator.predict(X_test_proba_merge)
    print("1: {}, 2:{}, 3:{}".format(sum(y_pred_test[y_pred_test==1]), 
          sum(y_pred_test[y_pred_test==2])/2, sum(y_pred_test[y_pred_test==3])/3))
    test_csv = '/newdisk/AVEC2018/downloaded_dataset/test_metadat.csv'
    test_df = pd.read_csv(test_csv)
    test_df.loc[:,'Predicted_label'] = y_pred_test
    test_df = test_df.loc[:, ['Instance_name', 'Predicted_label']]    
    des = 'BDS_NISL_5.csv'
    test_df.to_csv(des, index=False, sep=',')

def feature_importance():
    feature_importances = []
    fig = plt.figure()
    plt.style.use('ggplot')
    plt.tight_layout()
    results = [pickle.load(open('results_201807021500.pkl','rb'))]

    for result in results:
        top2_results = sort_results(results=result)[-2:]
        proba_list = []
        for result in top2_results:
            #
            feature_list = result[0]
            cv_dict = result[1]
            classifier = cv_dict['classifier']
            best_params = cv_dict['best_params']
            print(result)
            # get keys
            dfs = []
            for feat in feature_list:
                csv_file = os.path.join(wk_dir, feat+'.csv')
                df = pd.read_csv(csv_file, skipinitialspace=True)
                dfs.append(df)
            feat_imp_count = {feat_name: [] for feat_name in feature_list}
            
            dataset = load_dataset(feature_list)
            X_train,y_train,_, feature_columns = dataset['train']
            X_dev, y_dev,_ = dataset['dev']
            X_test = dataset['test']
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_dev = scaler.transform(X_dev)
            X_test = scaler.transform(X_test)
    
            lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
            model = SelectFromModel(lsvc, prefit=True)
            num_f_bef = X_train.shape
            X_train = model.transform(X_train)
            X_dev = model.transform(X_dev)
            X_test = model.transform(X_test)
            
            print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))
            # identify which features have been removed, which features have been saved
            selection = model.get_support()
            selected_index  = list(np.where(selection==True)[0])
            feature_columns = feature_columns[selected_index]
            if classifier=='NB':
                estimator = MultinomialNB()
            elif classifier=='SVC':
                estimator = SVC(probability=True)
            elif classifier=='RF':
                estimator = RandomForestClassifier(random_state=220)
            
            estimator.set_params(**best_params)
            
            # train
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_dev)
            print("Average recall: {}".format(recall_score(y_dev, y_pred, average='micro')))
            # predict probabilities as features
            X_train_proba = estimator.predict_proba(X_train)
            X_dev_proba = estimator.predict_proba(X_dev)
            X_test_proba = estimator.predict_proba(X_test)
            
            proba_list.append([X_train_proba, X_dev_proba, X_test_proba])
            
            #feature importance
            if classifier=='RF':
                feature_importance = estimator.feature_importances_
                feat_imp_df = pd.DataFrame(feature_importance, index = feature_columns, 
                                           columns =['importance']).sort_values('importance', ascending=False)
                feature_importances.append(feat_imp_df)
                # count feature num
                print(feat_imp_df[:100])
                for index, row in feat_imp_df[:100].iterrows():
                    importance = row['importance']
                    name = index
                    flag = 0
                    for i in range(len(feature_list)):
                        keys = dfs[i].keys()
                        if name in keys:
                            flag=1
                            break
                    assert flag==1
                    feature_name = feature_list[i]
                    feat_imp_count[feature_name].append(importance)
                name_list = []
                num_list = []
                count_list = []
                sum_list = []
                for key in feat_imp_count.keys():
                    if not len(feat_imp_count[key])==0:
                        imp_list = feat_imp_count[key]
                        aver_imp = np.mean(imp_list)
                        summation = sum(imp_list)
                        count = len(imp_list)
                        name_list.append(key)
                        num_list.append(aver_imp)
                        count_list.append(count)
                        sum_list.append(summation)

                sum_list, name_list, count_list = zip(*sorted(zip(sum_list, name_list, count_list)))
                #name_list = ['facial_lmk' if x=='face' else x for x in name_list]
                name_list = ['eye gaze' if x=='gaze_angle' else x for x in name_list]
                name_list = ['eGeMAPS' if x=='egmaps' else x for x in name_list]
                name_list = ['MFCC' if x=='mfcc' else x for x in name_list]
                name_list = ['body pose' if x=='pose' else x for x in name_list]

                ax = plt.subplot(111)
                ax.set_color_cycle(sns.color_palette("coolwarm_r", len(name_list)))
                ax.axes.tick_params(axis='y', labelsize=45, color='black')
                ax.axes.tick_params(axis='x', labelsize=45, color='black')
                plt.barh(np.arange(len(sum_list)), sum_list, 0.7, tick_label = name_list)
#                for i, v in enumerate(num_list):
#                    plt.text(v-0.01, i, str(count_list[i]), color='gray', fontweight='bold')
                plt.title('RF Classifier ', fontsize=45)
                plt.xlabel('Importance', fontsize=45)

    fig.set_size_inches(30,18)
    fig.savefig('feature_importance.png', dpi=100)
def remove_test():
    results = pickle.load(open('results_201807021500.pkl','rb'))

    example = ('pose', 'gaze_angle', 'FAUs', 'egmaps', 'mfcc','speech')
    example_resutls = [res for res in results if res[0]==example ]

    print(example_resutls)
    example = ('pose', 'gaze_angle', 'FAUs', 'mfcc', 'speech')
    egemaps_remove = [res for res in results if res[0]==example ]

    print(egemaps_remove)
    example = ('pose', 'gaze_angle', 'FAUs', 'egmaps', 'speech')
    mfcc_remove = [res for res in results if res[0]==example ]

    print(mfcc_remove)
    example = ('pose', 'FAUs', 'egmaps', 'mfcc', 'speech')
    eye_remove = [res for res in results if res[0]==example ]

    print(eye_remove)
    example = ('gaze_angle',  'FAUs', 'egmaps', 'mfcc', 'speech')
    pose_remove = [res for res in results if res[0]==example ]

    print(pose_remove)
    
    example = ('pose', 'gaze_angle', 'egmaps', 'mfcc', 'speech')
    fau_remove = [res for res in results if res[0]==example ]
    print(fau_remove)
def test_feature_selection():
    sys.path.append('/home/ddeng/feature-selector')
    from feature_selector import FeatureSelector
    df = pd.read_csv('hand_crafted_features_va.csv') 
    
    train_df = df.loc[df['Partition']=='train']
    train_df.reset_index(drop =True, inplace=True)
    train_n_sample = len(train_df)
    dev_df = df.loc[df['Partition']=='dev']
    dev_df.reset_index(drop =True, inplace=True)
    dev_n_sample = len(dev_df)
    
    #train 
    y_train = train_df['ManiaLevel']
    train_df = train_df.drop(columns = ['Instance_name', 'SubjectID','Age', 'Total_YMRS', 'ManiaLevel','Partition'])
    y_dev = dev_df['ManiaLevel']
    dev_df = dev_df.drop(columns = ['Instance_name', 'SubjectID','Age', 'Total_YMRS', 'ManiaLevel','Partition'])
    #selection 
    fs = FeatureSelector(data = train_df, labels = y_train)
    fs.identify_missing(missing_threshold=0.6)
    miss_features = fs.ops['missing']
    print(miss_features[:10])
    print(fs.missing_stats.head(10))
    
    fs.identify_single_unique()
    single_unique = fs.ops['single_unique']
    print(single_unique)
    fs.unique_stats.sample(5)
    

    #highly correlated features
    fs.identify_collinear(correlation_threshold=0.98)
    correlated_features = fs.ops['collinear']
    print(correlated_features[:5])
    
    #zero importance
    fs.identify_zero_importance(task = 'classification', eval_metric='multi_error',
                                n_iterations=10, early_stopping=True)
    
    one_hot_features = fs.one_hot_features
    base_features = fs.base_features
    print('There are %d original features' % len(base_features))
    print('There are %d one-hot features' % len(one_hot_features))
    
    fs.data_all.head(10)
    
    zero_importance_features = fs.ops['zero_importance']
    print(zero_importance_features[10:15])
    
    fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
    fs.identify_low_importance(cumulative_importance = 0.99)
    
    low_importance_features = fs.ops['low_importance']
    low_importance_features[:5]
    
    train_no_missing = fs.remove(methods = ['missing'])

    train_no_missing_zero = fs.remove(methods = ['missing', 'zero_importance'])

    all_to_remove = fs.check_removal()
    print(all_to_remove[10:25])
    
    train_removed = fs.remove(methods = 'all')
    train_removed_all = fs.remove(methods = 'all', keep_one_hot=False)
    train_removed_all = train_removed_all.drop(columns=['Gender'])
    print('Original Number of Features', train_df.shape[1])
    print('Final Number of Features: ', train_removed_all.shape[1])
    
    #dev 
    all_to_remove.remove('Gender_M')
    all_to_remove.remove('Gender_F')
    all_to_remove.append('Gender')
    dev_removed_all = dev_df.drop(columns = all_to_remove)
    print('Original Number of Features', dev_df.shape[1])
    print('Final Number of Features: ', dev_removed_all.shape[1])
    
    return [train_removed_all.values, y_train.values], [dev_removed_all.values, y_dev.values]
[X_train,y_train],[X_dev, y_dev] = test_feature_selection()
X_train = preprocessing.scale(X_train)
X_dev = preprocessing.scale(X_dev)
print(cv_on_RF(X_train, y_train, X_dev, y_dev))
print(cv_on_SVC(X_train, y_train, X_dev, y_dev))
    