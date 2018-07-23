#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:11:20 2018
bi-rnn using keras
@author: ddeng
"""
#
#from openface_provider import openface_provider
#from soundnet_provider import soundnet_provider
import pandas as pd
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD
import pdb
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import math
from sklearn.metrics import recall_score
from sklearn.model_selection import ParameterGrid
from keras.regularizers import L1L2
import pickle
from tqdm import tqdm
np.random.seed(220)
metadata_path = '/newdisk/AVEC2018/downloaded_dataset/labels_metadata.csv'
test_meta_path = '/newdisk/AVEC2018/downloaded_dataset/test_metadat.csv'
opface_dir = ['/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/vgg16_fc6',
              '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/vgg16_fc7']
sou_dir = ['/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/soundnet_conv7',
            '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/soundnet_pool5',
           '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/soundnet_conv5']
deep_dir = '.deep_spectrum'
def load_my_data():
    df = pd.read_csv(metadata_path)
    # load train and dev set
    
    v_data = {'train':[], 'dev':[]}
    v_target = {'train':[], 'dev':[]}
    a_data = {'train':[], 'dev':[]}
    a_target = {'train':[], 'dev':[]}
    for index, row in df.iterrows():
        video_name = df.loc[index, 'Instance_name']
        partition = df.loc[index, 'Partition']
        label = df.loc[index, 'ManiaLevel']
        sound_feature = np.load(os.path.join(sou_dir, video_name+'.npy'))
        opface_feature = np.load(os.path.join(opface_dir, video_name+'.npy'))

        v_data[partition].append(opface_feature)
        v_target[partition].append(label)
        a_data[partition].append(sound_feature)
        a_target[partition].append(label)

    a_data['train'] = np.asarray(a_data['train'])
    a_data['dev'] = np.asarray(a_data['dev'])
    v_data['train'] = np.asarray(v_data['train'])
    v_data['dev'] = np.asarray(v_data['dev'])
    a_target['train'] = np.asarray(a_target['train'])
    a_target['dev'] = np.asarray(a_target['dev'])
    v_target['train'] = np.asarray(v_target['train'])
    v_target['dev'] = np.asarray(v_target['dev'])
    
    return [[a_data, a_target], [v_data, v_target]]
def load_my_data_test():
    df = pd.read_csv(test_meta_path)
    # load train and dev set
    
    v_data = {'test':[]}
    a_data = {'test':[]}
    for index, row in df.iterrows():
        video_name = df.loc[index, 'Instance_name']
        sound_feature = np.load(os.path.join(sou_dir, video_name+'.npy'))
        opface_feature = np.load(os.path.join(opface_dir, video_name+'.npy'))

        v_data['test'].append(opface_feature)

        a_data['test'].append(sound_feature)


    a_data['test'] = np.asarray(a_data['test'])

    v_data['test'] = np.asarray(v_data['test'])
    
    return [a_data, v_data]
def load_my_data_deep():
    df = pd.read_csv(metadata_path)
    # load train and dev set
    a_data = {'train':[], 'dev':[]}
    a_target = {'train':[], 'dev':[]}
    for index, row in df.iterrows():
        video_name = df.loc[index, 'Instance_name']
        partition = df.loc[index, 'Partition']
        label = df.loc[index, 'ManiaLevel']
        sound_feature = np.load(os.path.join(deep_dir, video_name+'.npy'))

        a_data[partition].append(sound_feature)
        a_target[partition].append(label)

    a_data['train'] = np.asarray(a_data['train'])
    a_data['dev'] = np.asarray(a_data['dev'])
    a_target['train'] = np.asarray(a_target['train'])
    a_target['dev'] = np.asarray(a_target['dev'])
    return [a_data, a_target]

def run_soundnet():
    #dev =0.6
    print('Loading data...')
    audio_data, video_data = load_my_data()
    #X_train, y_train = np.concatenate((audio_data[0]['train'], video_data[0]['train'][:,:10,:]), axis=-1), audio_data[1]['train']
    #split_frac = 0.5
    #split_index = int(split_frac * len(data['dev']))
    #X_val, y_val = np.concatenate((audio_data[0]['dev'], video_data[0]['dev'][:,:10,:]), axis=-1), audio_data[1]['dev']
    X_train, y_train = video_data[0]['train'], video_data[1]['train']
    X_val, y_val = video_data[0]['dev'], video_data[1]['dev']
    # transform label
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_train = to_categorical(y_train)
    y_val = encoder.transform(y_val)
    y_val = to_categorical(y_val)
    #y_test = encoder.transform(y_test)
    #y_test = to_categorical(y_test)
    print('train set shape: {}\n val set shape:{} '.format(X_train.shape, X_val.shape))
    print('train target shape: {}\n val target shape:{} '.format(y_train.shape, y_val.shape))
    time_steps = X_train.shape[1]
    input_dim = X_train.shape[2]
    batch_size=20
    hidden_size= 1024
    lstm_layers=2
    keep_prob=0.5
    class_num=3
    # save the model
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    checkpointer = ModelCheckpoint(
            filepath = os.path.join('checkpoints', 'dev_best_fc7.hdf5' ),
            monitor = 'val_acc', 
            verbose=1,
            save_best_only=True)
    def step_decay(epoch):
        initial_lrate = 0.01
        drop = 0.5
        epochs_drop=20
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    lrate = LearningRateScheduler(step_decay)
    
    # bi-directional LSTM
    
    model = Sequential()
    #model.add(TimeDistributed(Dense(input_dim//3, activation='relu') , input_shape = (time_steps, input_dim)))
    #model.add(Dropout(keep_prob))
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=False), input_shape = (time_steps, input_dim)))
    model.add(Dropout(keep_prob))
    model.add(Dense(class_num, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
    
    print('Train...')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=100,
              validation_data=[X_val, y_val], 
              callbacks = [checkpointer, lrate])
    
    print('Testing ...')
    class_names = ['remission','hypomania','mania']
    filepath = os.path.join('checkpoints','dev_best_fc7.hdf5')
    best_model = load_model(filepath)
    y_pred = best_model.predict(X_val)
    print(classification_report(y_val.argmax(axis=-1), y_pred.argmax(axis=-1), target_names=class_names))
def run_deep():
    #dev=0.5
    print('Loading data...')
    data, target = load_my_data_deep()
    X_train, y_train = data['train'], target['train']
    X_val, y_val = data['dev'], target['dev']
    # transform label
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_train = to_categorical(y_train)
    y_val = encoder.transform(y_val)
    y_val = to_categorical(y_val)
    #y_test = encoder.transform(y_test)
    #y_test = to_categorical(y_test)
    print('train set shape: {}\n val set shape:{} '.format(X_train.shape, X_val.shape))
    print('train target shape: {}\n val target shape:{} '.format(y_train.shape, y_val.shape))
    time_steps = X_train.shape[1]
    input_dim = X_train.shape[2]
    batch_size=20
    hidden_size= 128
    keep_prob=0.5
    class_num=3
    # save the model
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    checkpointer = ModelCheckpoint(
            filepath = os.path.join('checkpoints', 'dev_best_deep.hdf5' ),
            monitor = 'val_acc', 
            verbose=1,
            save_best_only=True)
    def step_decay(epoch):
        initial_lrate = 0.01
        drop = 0.5
        epochs_drop=20
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    lrate = LearningRateScheduler(step_decay)
    
    # bi-directional LSTM
    
    model = Sequential()

    model.add(Bidirectional(LSTM(hidden_size, return_sequences=False), input_shape = (time_steps, input_dim)))
    model.add(Dropout(keep_prob))
    model.add(Dense(class_num, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
    
    print('Train...')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=100,
              validation_data=[X_val, y_val], 
              callbacks = [checkpointer, lrate])
    
    print('Testing ...')
    class_names = ['remission','hypomania','mania']
    filepath = os.path.join('checkpoints', 'dev_best_deep.hdf5')
    best_model = load_model(filepath)
    y_pred = best_model.predict(X_val)
    print(classification_report(y_val.argmax(axis=-1), y_pred.argmax(axis=-1), target_names=class_names))
def save_test():
    best_model_path = os.path.join('checkpoints', 'dev_best_pool5.hdf5')

    best_model = load_model(best_model_path)
    audio_data,_ = load_my_data_test()
    X_test = audio_data['test']
    pred = best_model.predict(X_test)
    pred = np.argmax(pred, axis=-1) + 1
    print('1: {}, 2:{}, 3:{}'.format(sum(pred[pred==1]), sum(pred[pred==2])/2, sum(pred[pred==3])/3))
    test_df = pd.read_csv(test_meta_path)
    test_df.loc[:,'Predicted_label'] = pred
    test_df = test_df.loc[:, ['Instance_name', 'Predicted_label']]    
#    des = 'BDS_NISL_2.csv'
#    test_df.to_csv(des, index=False, sep=',')
def dataloader(dataset_path):
    df = pd.read_csv(metadata_path)
    # load train and dev set

    data = {'train': [], 'dev': []}
    target = {'train': [], 'dev': []}
    for index, row in df.iterrows():
        video_name = df.loc[index, 'Instance_name']
        partition = df.loc[index, 'Partition']
        label = df.loc[index, 'ManiaLevel']
        feature = np.load(os.path.join(dataset_path, video_name + '.npy'))
        data[partition].append(feature)
        target[partition].append(label)

    data['train'] = np.asarray(data['train'])
    data['dev'] = np.asarray(data['dev'])
    target['train'] = np.asarray(target['train'])
    target['dev'] = np.asarray(target['dev'])
    return [data, target]
def experiment(**arg):

    dataset_path = arg['dataset_path']
    n_neurons = arg['n_neurons']
    drop_ratio = arg['drop_ratio']
    n_epochs = arg['n_epochs']
    n_batch = arg['n_batch']
    regularizaton = arg['regularization']
    ini_lrrate = arg['ini_lrrate']

    data, target = dataloader(dataset_path)
    type_of_data = dataset_path.split('/')[-1]
    X_train, y_train = data['train'], target['train']
    X_dev, y_dev = data['dev'], target['dev']
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_train = to_categorical(y_train)
    y_dev = encoder.transform(y_dev)
    y_dev= to_categorical(y_dev)

    print('train set shape: {}\n val set shape:{} '.format(X_train.shape, X_dev.shape))
    print('train target shape: {}\n val target shape:{} '.format(y_train.shape, y_dev.shape))
    time_steps = X_train.shape[1]
    input_dim = X_train.shape[2]

    batch_size= n_batch
    hidden_size= n_neurons
    drop_ratio = drop_ratio
    class_num=3
    epochs = n_epochs
    # save the model
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(os.path.join('checkpoints', type_of_data)):
        os.makedirs(os.path.join('checkpoints', type_of_data))
    regul_name = 'l1 %.2f,l2 %.2f' % (regularizaton.l1, regularizaton.l2)
    model_path = os.path.join('checkpoints', type_of_data, 'neuron_{}_drop_{}_epoch_{}_batch_{}_regul_{}_lr_{}.hdf5'.format(
                n_neurons, drop_ratio, n_epochs, n_batch, regul_name, ini_lrrate))
    checkpointer = ModelCheckpoint(
            filepath = model_path,
            monitor = 'val_acc',
            verbose=1,
            save_best_only=True)

    def step_decay(epoch):
        initial_lrate = ini_lrrate
        drop = 0.5
        epochs_drop=20
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    lrate = LearningRateScheduler(step_decay)

    # bi-directional LSTM

    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=False, kernel_regularizer=regularizaton), input_shape=(time_steps, input_dim)))
    model.add(Dropout(drop_ratio))
    model.add(Dense(class_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=ini_lrrate, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[X_dev, y_dev],
              callbacks=[checkpointer, lrate], verbose=0)

    print('Testing ...')

    best_model = load_model(model_path)
    y_pred = best_model.predict(X_dev)
    # score
    score_dev = recall_score(y_dev.argmax(axis=-1), y_pred.argmax(axis=-1), average='micro')
    if score_dev<0.55:
        os.remove(model_path)
        print("model removed")
    print(model_path+': '+str(score_dev))
    return [type_of_data, model_path, score_dev]


def grid_search_lstm():

    para_grid = {'dataset_path': sou_dir,  'n_neurons': [512, 256, 128],
                 'drop_ratio': [0.2, 0.5, 0.8], 'n_epochs': [100],
                 'n_batch': [20, 40],
                 'regularization': [ L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01),
                                    L1L2(l1=0.01, l2=0.01)],
                 'ini_lrrate': [0.1, 0.01, 0.005]}

    results = []
    params_grid = list(ParameterGrid(para_grid))
    for param in tqdm(params_grid):
        pass
        
        results.append(experiment(**param))
    with open('results_sound_0708.pkl', 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        
#    para_grid = {'dataset_path': opface_dir,  'n_neurons': [512, 256, 128],
#                 'drop_ratio': [0.2, 0.5, 0.8], 'n_epochs': [100],
#                 'n_batch': [ 20],
#                 'regularization': [ L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01),
#                                    L1L2(l1=0.01, l2=0.01)],
#                 'ini_lrrate': [0.1, 0.01, 0.005]}
#
#    results = []
#    params_grid = list(ParameterGrid(para_grid))
#    for param in tqdm(params_grid):
#        pass
#        
#        results.append(experiment(**param))
#    with open('results_visual_0708.pkl', 'wb') as file:
#        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

pdb.set_trace()
grid_search_lstm()