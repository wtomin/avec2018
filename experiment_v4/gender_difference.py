#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:09:54 2018
study on gender difference
@author: ddeng
"""
import pandas as pd
import pdb
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools
# Male patients ar more than Female patients
def load_gender_difference_data():
    df = pd.read_csv('hand_crafted_features_va.csv')
    Female_samples = df.loc[df['Gender']=='F']
    Female_samples.reset_index(inplace=True, drop=True)
    Male_samples = df.loc[df['Gender']=='M']
    Male_samples.reset_index(inplace=True, drop=True)
    n_sample_female = len(Female_samples)
    n_sample_male = len(Male_samples)
    n_feature = len(df.keys()[7:])
    data_female = np.empty((n_sample_female, n_feature), dtype = np.float64)
    data_male = np.empty((n_sample_male, n_feature), dtype = np.float64)
    target_female = np.empty((n_sample_female,), dtype=np.int)
    target_male = np.empty((n_sample_male, ), dtype=np.int)
    
    #female dataset
    for index, row in Female_samples.iterrows():
        data_female[index] = np.asarray(Female_samples.loc[index, Female_samples.keys()[7:]], dtype= np.float64)
        target_female[index] = np.asarray(Female_samples.loc[index,'ManiaLevel'], dtype=np.int)
    female_dict = {'data':data_female, 'target': target_female}
    #male dataset
    for index, row in Male_samples.iterrows():
        data_male[index] = np.asarray(Male_samples.loc[index, Male_samples.keys()[7:]], dtype= np.float64)
        target_male[index] = np.asarray(Male_samples.loc[index,'ManiaLevel'], dtype=np.int)
    male_dict = {'data':data_male, 'target': target_male}
    
    return {'female': female_dict, 'male': male_dict}
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
def gender_diff_cv_on_SVC():
    dataset = load_gender_difference_data()
    female_data, male_data = dataset['female'], dataset['male']
    
    #train a model for female samples first
    X,y = female_data['data'], female_data['target']
    X = preprocessing.scale(X)
    print("Female samples are :{}".format(X.shape[0]))
    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.3, random_state=12, shuffle=True, stratify=y)
    ##############################################
    ticks = time.time()
    
    estimator = SVC(kernel='linear')

    lsvc = LinearSVC(C=10, dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))

    # classifications
    param_grid = [
  {'C': [ 1, 10, 100, 1000, 10000, 1e5], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7], 'kernel': ['rbf']},
]
    score = 'recall'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s_macro' % score)
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
    ###################################################
    X,y = male_data['data'], male_data['target']
    X = preprocessing.scale(X)
    print("Male samples are :{}".format(X.shape[0]))
    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.3, random_state=12, shuffle=True, stratify=y)
    ticks = time.time()
    
    estimator = SVC(kernel='linear')

    lsvc = LinearSVC(C=10, dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))

    # classifications
    param_grid = [
  {'C': [ 1, 10, 100, 1000, 10000, 1e5], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7], 'kernel': ['rbf']},
]
    score = 'recall'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s_macro' % score)
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
    ###################################################
    
def gender_diff_cv_on_RF():
    dataset = load_gender_difference_data()
    female_data, male_data = dataset['female'], dataset['male']
    #train a model for female samples first
    X,y = female_data['data'], female_data['target']
    X = preprocessing.scale(X)
    print("Female samples are :{}".format(X.shape[0]))
    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.3, random_state=12, shuffle=True, stratify=y)
    # feature normalization

    ticks = time.time()
    
    estimator = RandomForestClassifier(random_state=49, n_jobs=8,oob_score=True)

    # classifications
    param_grid = { 
        'n_estimators': [400, 500, 600],
        'max_features': ['auto'],
        'max_depth' : [10,20],
        'criterion' :['gini', 'entropy']
    }
    score = 'recall'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s_macro' % score)
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
    #################################################################
    X,y = male_data['data'], male_data['target']
    X = preprocessing.scale(X)
    print("Male samples are :{}".format(X.shape[0]))
    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.3, random_state=12, shuffle=True, stratify=y)
    # feature normalization

    ticks = time.time()
    
    estimator = RandomForestClassifier(random_state=49, n_jobs=8,oob_score=True)

    # classifications
    param_grid = { 
        'n_estimators': [400, 500, 600],
        'max_features': ['auto'],
        'max_depth' : [10,20],
        'criterion' :['gini', 'entropy']
    }
    score = 'recall'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s_macro' % score)
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
if __name__ == "__main__":

    pdb.set_trace()
    gender_diff_cv_on_RF()