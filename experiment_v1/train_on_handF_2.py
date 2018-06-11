#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:21:35 2018

@author: ddeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pdb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
import itertools
from keras.optimizers import SGD
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
    for index, row in train_df.iterrows():
        data[index] = np.asarray(train_df.loc[index, train_df.keys()[7:]], dtype =np.float64)
        target[index] = np.asarray(train_df.loc[index, 'ManiaLevel'])
    train_set = [data, target]
    data = np.empty((dev_n_sample, n_feature) , dtype = np.float64)
    target = np.empty((dev_n_sample, ), dtype = np.int)
    for index, row in dev_df.iterrows():
        data[index] = np.asarray(dev_df.loc[index, dev_df.keys()[7:]], dtype =np.float64)
        target[index] = np.asarray(dev_df.loc[index, 'ManiaLevel'])
    dev_set = [ data, target]
    return full_dict, {'train':train_set, 'dev': dev_set}

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
def main():
    dataset, _= load_my_data()
    X, y = dataset['data'], dataset['target']
    X = preprocessing.scale(X)
    # bianry classification
    y[y==3]=2
    # split for cross validation
    #random_state=409, 
#    stratify = np.zeros((X.shape[0]//2, ))
#    stratify = np.concatenate((stratify, np.ones(X.shape[0] - X.shape[0]//2, )))

    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.4, random_state=10, shuffle=True, stratify=y)
    # feature normalization

    ticks = time.time()
    
    estimator = SVC(kernel='linear')
#    selector = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5),
#                  scoring ='recall_macro')
#    selector.fit(X_train, y_train)
#    num_f_bef  = X_train.shape
#    #transform on two sets
#    X_train = selector.transform(X_train)
#    X_test = selector.transform(X_test)
    lsvc = LinearSVC(C=10, dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))

    # classifications
    param_grid = [
  {'C': [ 1, 10, 100, 1000, 10000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7], 'kernel': ['rbf']},
      ]
    score = 'recall'
    class_names = ['remission','mania']
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

def create_model(dropout_rate=0.0, neurons=300, optimizer='adam', learn_rate=0.01, momentum=0.9):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=1535, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    optimizer = SGD(lr = learn_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
def cv_on_deep_nn():
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    dataset, _= load_my_data()
    X, y = dataset['data'], dataset['target']
#    encoder = LabelEncoder()
#    encoder.fit(y)
#    y = encoder.transform(y)
#    y = to_categorical(y)
    X = preprocessing.scale(X)
    #random_state=409, 
    stratify = np.zeros((X.shape[0]//3, ))
    stratify = np.concatenate((stratify, np.ones(X.shape[0]//3, )))
    stratify = np.concatenate((stratify, np.ones(X.shape[0] - 2*(X.shape[0]//3), )+1))
    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.4, random_state=409, shuffle=True, stratify=stratify)
    score = 'recall'
    class_names = ['remission','hypomania','mania']
    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)
    # define the grid search parameters
#    batch_size = [10, 20, 40, 60, 80, 100]
#    epochs = [10, 50, 100]
    batch_size = [20, 40]
    epochs = [ 50, 100]
    dropout_rate = [ 0.3, 0.5, 0.6, 0.7]
    neurons = [300, 600]
    learn_rate = [0.001, 0.01]
    momentum = [0.0,0.3, 0.6, 0.8, 0.9]
    param_grid = dict(batch_size=batch_size, epochs=epochs,
                      neurons=neurons, dropout_rate = dropout_rate,
                      learn_rate=learn_rate, momentum=momentum)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='%s_macro' % score, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    #test
    y_true, y_pred = y_test, grid.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()

    # confusion matrix

    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names,
                          title = 'Confusion matrix without normalization')
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names, normalize=True,
                          title = 'Confusion matrix')  
    print(classification_report(y_test, y_pred))
    
if __name__ == "__main__":

    pdb.set_trace()
    main()