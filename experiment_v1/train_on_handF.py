#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:08:16 2018
using cross validation and recurrsive feature selection method
@author: ddeng
"""
import pandas as pd
from sklearn.datasets.base import Bunch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.feature_selection import SelectFromModel
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import itertools
def load_my_data():
    df = pd.read_csv('hand_crafted_features_va.csv') 
    n_sample = len(df['Instance_name'])
    n_feature = len(df.keys()[7:])
    data = np.empty((n_sample, n_feature) , dtype = np.float64)
    target = np.empty((n_sample, ), dtype = np.int)
    
    for index, row in df.iterrows():
        data[index] = np.asarray(df.loc[index, df.keys()[7:]], dtype =np.float64)
        target[index] = np.asarray(df.loc[index, 'ManiaLevel'])
        
    return Bunch(data=data, target=target)
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

def main():
    dataset = load_my_data()
    X = dataset.data
    y = dataset.target - 1
    # split

    
    # feature normalization
    X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.33, random_state=42)
    # Create the RFE object and compute a cross-validated score.
    clf = SVC(kernel ='linear')
    # The "accuracy" scoring is proportional to the number of correct
    ticks = time.time()
    # classifications
    
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5),
                  scoring ='recall_macro')
    rfecv.fit(X_train, y_train)
    print('Time Elapse: {}'.format(time.time()-ticks))
    print("Optimal number of features : %d" % rfecv.n_features_)


    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    

    # confusion matrix
    y_pred = rfecv.predict(X_test)
    class_names = ['remission','hypomania','mania']
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names,
                          title = 'Confusion matrix without normalization')
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names, normalize=True,
                          title = 'Confusion matrix')  

if __name__ == "__main__":

    main()