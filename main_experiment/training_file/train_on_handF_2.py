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
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge, RidgeCV
import itertools
import subprocess
import os
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
    dev_set = [ data, target, target_y]
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
def cv_on_RF():
    dataset, _= load_my_data()
    X, y = dataset['data'], dataset['target']
    X = preprocessing.scale(X)
    # bianry classification
#    y[y==3]=2
    # split for cross validation
    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.4, random_state=409, shuffle=True, stratify=y)
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

    line = "Best: %f using %s" % (clf.best_score_, clf.best_params_)
    return [classification_report(y_true, y_pred, target_names=class_names),line]
def train_on_NB():
    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    from sklearn import tree
    dataset, _= load_my_data()
    X, y = dataset['data'], dataset['target']
    X = preprocessing.scale(X)
    # bianry classification
#    y[y==3]=2
    # split for cross validation
    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.4, random_state=10, shuffle=True, stratify=y)
    # feature normalization

    ticks = time.time()
    gnb =  GaussianNB()
    gnb.fit(X_train, y_train)
    print('Time Elapse: {}'.format(time.time()-ticks))
    y_true, y_pred = y_test, gnb.predict(X_test)
    class_names = ['remission','hypomania','mania']
    print(classification_report(y_true, y_pred, target_names=class_names))
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names, normalize=True,
                          title = 'Confusion matrix') 
def cv_on_SVC():
    _, dataset= load_my_data()
    X_train,y_train,_ = dataset['train']
    X_test, y_test,_ = dataset['dev']
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    # bianry classification
#    y[y==3]=2
    # feature normalization

    ticks = time.time()
    
    estimator = Perceptron(n_iter=50)

    lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))

    # classifications
#    param_grid = [
#  {'C': [ 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
#         0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 10, 100, 1000, 10000, 1e5], 'kernel': ['linear']},
#    {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7], 'kernel': ['rbf']},
#]
    param_grid = {'n_iter':[10, 20, 30, 40, 50, 60, 70]}
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

def create_model(input_dim=1596, dropout_rate=0.0, neurons=300, optimizer='adam', learn_rate=0.01, momentum=0.9):
    # create model
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.utils import to_categorical
    from keras.optimizers import SGD
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    optimizer = SGD(lr = learn_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
def cv_on_deep_nn():
    # fix random seed for reproducibility
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.utils import to_categorical
    from keras.optimizers import SGD
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

    X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size = 0.4, random_state=409, shuffle=True, stratify=y)
    score = 'recall'
    class_names = ['remission','hypomania','mania']
    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)
    # define the grid search parameters
#    batch_size = [10, 20, 40, 60, 80, 100]
#    epochs = [10, 50, 100]
    input_dim = [X_train.shape[1]]
    batch_size = [20, 40, 60, 80]
    epochs = [100, 200]
    dropout_rate = [ 0.5, 0.6,0.7, 0.8]
    neurons = [ 300, 600, 800, 1200]
    learn_rate = [0.01,0.001]
    momentum = [ 0.6, 0.7, 0.8, 0.9]
    param_grid = dict(input_dim=input_dim,batch_size=batch_size, epochs=epochs,
                      neurons=neurons, dropout_rate = dropout_rate,
                      learn_rate=learn_rate, momentum=momentum)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='%s_micro' % score)
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
#
#    plt.figure()
#    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names,
#                          title = 'Confusion matrix without normalization')
#    plt.figure()
#    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names, normalize=True,
#                          title = 'Confusion matrix')  
    best_param = "Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)
    return classification_report(y_true, y_pred, target_names=class_names), best_param
def generate_feature_combination(features_list):
    feature_combinations = []
    cmds = []
    # no deletion
    feature_combinations.append(features_list)
    cmds.append(['python', os.path.join(os.getcwd(),'hand_crafted_features.py')])
    #delete one
    for feat in features_list:
        copy = list(features_list)
        copy.remove(feat)
        feature_combinations.append(copy)
        cmds.append(['python',os.path.join(os.getcwd(),'hand_crafted_features.py'), '--'+feat])
    #delete two
    for feat1 in features_list:
        copy2 = list(features_list)
        copy2.remove(feat1)
        for feat2 in copy2:
            copy2.remove(feat2)
            feature_combinations.append(copy2)
            cmds.append(['hand_crafted_features.py', '--'+feat1, '--'+feat2])
            
    return feature_combinations, cmds
def grid_search_on_feature_set():
    # grid search on features
    features = ['head', 'au','gaze','pose','lld_audio','word']
    text_file = open('Output.txt','w')
    # delete at most two features
    feature_combinations, cmds = generate_feature_combination(features)
    for feat_combi, cmd in zip(feature_combinations, cmds):
        text_file.write("#################################################\nFeature Combinations:%s\n"%" ".join(feat_combi))
        subprocess.call(cmd)
        print('Feature extraction done.')
        class_report, line = cv_on_SVC()
        text_file.write(line+'\n')
        text_file.write(class_report)
        text_file.write('\n')
    text_file.close()
def cv_on_SVM_regression():
    _, dataset= load_my_data()
    X_train,_,y_train = dataset['train']
    X_test, _, y_test = dataset['dev']
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    
    ticks = time.time()
    
    estimator = SVR()

    lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))

    # classifications
    param_grid = [
  {'C': [ 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
         0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 10, 100, 1000, 10000, 1e5], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7], 'kernel': ['rbf']},
]
    score = 'neg_mean_squared_error'
    class_names = ['remission','hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=2, scoring=score)
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
#    rcv = RidgeCV(alphas=np.arange(1300, 1320, 1))
#    rcv.fit(X_train, y_train)
#    print(rcv.alpha_)
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
def cv_on_SVC_eval_on_dev():
    _, dataset= load_my_data()
    X_train,y_train,_ = dataset['train']
    X_test, y_test,_ = dataset['dev']
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    # bianry classification
#    y[y==3]=2
    # feature normalization

    ticks = time.time()
    
    estimator =SVC(kernel='linear')

    lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))

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
    best_score = 0
    for g in ParameterGrid(param_grid):
        estimator.set_params(**g)
        estimator.fit(X_train,y_train) 
        y_true, y_pred = y_test, estimator.predict(X_test)
        score = recall_score(y_true, y_pred, average='micro')
        if score>best_score:
            best_score=score
            best_param = g
    print('Time Elapse: {}'.format(time.time()-ticks))
    
    print("Best parameters set found on development set:")
    print()
    print(best_param)
    print()
    print("Best scores found on development set:")
    print()
    print(best_score)
    print("Grid scores on development set:")
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    estimator = SVC()
    estimator.set_params(**best_param)
    estimator.fit(X_train, y_train)
    y_true, y_pred = y_test, estimator.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()

    # confusion matrix

    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names,
                          title = 'Confusion matrix without normalization')
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=class_names, normalize=True,
                          title = 'Confusion matrix')  

def cv_binary_regression():
    _, dataset= load_my_data()
    X_train,y_train_m,y_train_y = dataset['train']
    X_test, y_test_m, y_test_y = dataset['dev']
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    
    #do binary classification using SVC
    y_train_m_binary = np.asarray(y_train_m.copy())
    y_train_m_binary[y_train_m_binary==3]=2
    y_test_m_binary = np.asarray(y_test_m.copy())
    y_test_m_binary[y_test_m_binary==3]=2
    
    #binary classification
    param_grid = [
  {'C': [ 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
         0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 10, 100, 1000, 10000, 1e5]}#, 'kernel': ['linear']},
#    {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7], 'kernel': ['rbf']},
]
    score = 'recall'
    class_names = ['remission','mania']
    
    ticks = time.time()
    
    estimator = LinearSVC()
    lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train_m)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=5, scoring='%s_micro' % score)
    clf.fit(X_train, y_train_m_binary)
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
    y_true, y_pred = y_test_m_binary, clf.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()
    
    #regression
    idxes_train = np.where(y_train_m_binary ==2)[0]
    idxes_test = np.where(y_pred==2)[0]
    X_train = X_train[idxes_train]
    y_train_y = y_train_y[idxes_train]
    
    X_test =X_test[idxes_test]
    y_test_y = y_test_y[idxes_test]
    
    ticks = time.time()
    estimator = Ridge()

    lsvc = LinearSVC(C=1e-2, dual=True).fit(X_train, y_train_y)
    model = SelectFromModel(lsvc, prefit=True)
    num_f_bef = X_train.shape
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    print('before selecting,{}; after selecting: {}'.format(num_f_bef[1], X_train.shape[1]))
    param_grid = {'alpha': np.arange(640, 670,0.1)}
    score = 'neg_mean_squared_error'
    class_names = ['hypomania','mania']
#    class_names = ['remission','mania']
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(estimator, param_grid, cv=6, scoring=score)
    clf.fit(X_train, y_train_y)
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
    for i in range(len(y_test_y)):
        y_true.append(get_class_from_ymrs(y_test_y[i]))
        y_pred.append(get_class_from_ymrs(pred[i]))
    print(classification_report(y_true, y_pred, target_names=class_names))
    print()  
    
if __name__ == "__main__":

    #pdb.set_trace()
    cv_binary_regression()
