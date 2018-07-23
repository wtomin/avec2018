#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:36:58 2018
hand crafted features
@author: ddeng
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from scipy.io.wavfile import read
from scipy.io import arff
import argparse
parser = argparse.ArgumentParser()  
parser.add_argument("--head", action="store_false",help="whether to add head features")  
parser.add_argument("--au" ,action="store_false", help="whether to add au features")  
parser.add_argument("--face" ,action="store_false", help="whether to add face features")  
parser.add_argument("--gaze", action="store_false" ,help="whether to add gaze features")
parser.add_argument("--pose", action="store_false", help="whether to add pose features")
parser.add_argument("--egmaps", action="store_false", help="whether to add eGeMaps lld audio features")
parser.add_argument("--mfcc", action="store_false", help="whether to add MFCC lld audio features")
parser.add_argument("--emo_large", action="store_false", help="whether to add Emolarge lld audio features")
parser.add_argument("--emobase", action="store_false", help="whether to add emobase2010 lld audio features")
parser.add_argument("--word", action="store_false", help="whether to add word features")
parser.add_argument("--test", action="store_true", help="whether to generate test features")

args = parser.parse_args()  

class Hand_Craftor_visual():
    def __init__(self):
        self.rate = 30 # 30Hz smaple rate
        self.metadata_path = '../labels_metadata.csv'
        self.test_meta_path = '../test_metadat.csv'
        self.gaze_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/gaze_angle'
        self.pose_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/pose'
        self.au_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/FAUs'
        self.face_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/face'
        self.head_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/head'
        self.gaze_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/gaze_angle.csv'
        self.pose_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/pose.csv'
        self.au_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/FAUs.csv'
        self.face_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/face.csv'
        self.head_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/head.csv'
        self.feat_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/'
    def save_features(self):

        # in metadata, train_dev set are included
        metadata = pd.read_csv(self.metadata_path, skipinitialspace=True)
        if args.test:
            metadata = pd.read_csv(self.test_meta_path,  skipinitialspace=True)
        #features about head motion
        if args.head:
            print('Extracting for head...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                head_f = self.extract_head_features(instance_name)
                for f in head_f:
                    f_name, f_value = f
                    for f_name_ins, f_value_ins in zip(f_name, f_value):
                        feature.loc[index, f_name_ins] = f_value_ins
            des =self.head_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)
                
        #feature about au

        if args.au:
            print('Extracting for FAU...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                au_f = self.extract_au_features(instance_name)
                for f in au_f:
                    f_name, f_value = f
                    for f_name_ins, f_value_ins in zip(f_name, f_value):
                        feature.loc[index, f_name_ins] = f_value_ins
            des = self.au_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)
        # features about gaze, average fixation time, angle difference

        if args.gaze:
            print('Extracting for Gaze...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                gaze_f = self.extract_gaze_features(instance_name)
                for f in gaze_f:
                    f_name, f_value = f
                    for f_name_ins, f_value_ins in zip(f_name, f_value):
                        feature.loc[index, f_name_ins] = f_value_ins
            des = self.gaze_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)
        # then add pose features
        if args.pose:
            print('Extracting for pose...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                pose_f = self.extract_pose_features(instance_name)
                for f in pose_f:
                    f_name, f_value = f
                    for f_name_ins, f_value_ins in zip(f_name, f_value):
                        feature.loc[index, f_name_ins] = f_value_ins
            des = self.pose_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)
        # then add face features
        if args.face:
            print('Extracting for face...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                pose_f = self.extract_face_features(instance_name)
                for f in pose_f:
                    f_name, f_value = f
                    for f_name_ins, f_value_ins in zip(f_name, f_value):
                        feature.loc[index, f_name_ins] = f_value_ins
            des = self.face_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)

    def extract_face_features(self, instance_name):
        csv_file = os.path.join(self.face_dir, instance_name+'.csv')
        df = pd.read_csv(csv_file, skipinitialspace=True)
        keys = df.keys()
        features_all = []
        for key in keys:
            temp_signal = df.loc[:,key].values
            names, feature = self.apply_functionals(temp_signal)
            f_names = [key+'_'+n for n in names]
            features_all.append([f_names, feature])

        return features_all
    def extract_head_features(self, instance_name):
        csv_file = os.path.join(self.head_dir, instance_name+'.csv')
        df = pd.read_csv(csv_file, skipinitialspace=True)
        df = df.loc[:, df.keys()[1:]] # remove frameTime
        keys = df.keys()
        features_all = []
        for key in keys:
            temp_signal = df.loc[:,key].values
            names, feature = self.apply_functionals(temp_signal)
            f_names = [key+'_'+n for n in names]
            features_all.append([f_names, feature])

        return features_all
    def extract_gaze_features(self, instance_name):
        csv_file = os.path.join(self.gaze_dir, instance_name+'.csv')
        df = pd.read_csv(csv_file, skipinitialspace=True)
        df = df.loc[:, df.keys()[1:]] # remove frameTime
        
        angle_velocities = df['angle_velocity']
        fixation = df['fixation']
        dialation = df['dialation']
        # angle related features
        f_name_angle, f_angle  = self.apply_functionals(angle_velocities)
        f_name_angle = ['angle_'+name for name in f_name_angle]
        # dialation related features
        f_name_dia, f_dia  = self.apply_functionals(dialation)
        f_name_dia = ['dialation_'+name for name in f_name_dia]
        #fixation time
        fixation_flag = True if fixation[0]==1 else False
        true_counter=0
        fixation_times = []
        for i in range(len(fixation)):
            if fixation[i]==1 and fixation_flag==False:
                fixation_flag=True
                true_counter+=1
            elif fixation[i]==1 and fixation_flag==True:
                true_counter+=1
            elif fixation[i]==0 and fixation_flag==True:
                fixation_flag=False
                fixation_times.append(true_counter)
                true_counter=0
        #fixation related features
        #ignore samll fixation times(<100ms)
        fixation_times = np.array(fixation_times)
        fixation_times = fixation_times[fixation_times>2]
        f_name_fix, f_fix = self.apply_functionals(fixation_times)
        f_name_fix = ['fixation_'+name for name in f_name_fix]
        f_name_fix.append('fixation_changes')
        f_fix.append(fixation_times.shape[0])
        return [[f_name_angle, f_angle], [f_name_fix, f_fix], [f_name_dia, f_dia]]
    def extract_au_features(self,instance_name):
        csv_file = os.path.join(self.au_dir, instance_name+'.csv')
        df = pd.read_csv(csv_file, skipinitialspace=True)
        df = df.loc[:, df.keys()[1:]] # remove frameTime
        full_name = []
        full_f = []
        # intensity(0,5)
        f_name_cate = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r','AU07_r', 
                        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
        for f_name in f_name_cate:
            f_name_inten, f_inten = self.apply_functionals(df[f_name])
            f_name_inten = [f_name+'_'+name for name in f_name_inten]
            full_name.extend(f_name_inten)
            full_f.extend(f_inten)
        
        f_name_cate =[ 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 
                      'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 
                      'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
        for f_name in f_name_cate:
            f_name_pres, f_pres = self.functionals_for_au_presence(df[f_name])
            f_name_pres = [f_name+'_'+name for name in f_name_pres]
            full_name.extend(f_name_pres)
            full_f.extend(f_pres)
            
        return [[full_name, full_f]]
    def extract_pose_features(self, instance_name):
        # except 3,4,6,7, other joints only calculate it's std
        csv_file = os.path.join(self.pose_dir, instance_name+'.csv')
        df = pd.read_csv(csv_file, skipinitialspace=True)
        df = df.loc[:, df.keys()[1:]] # remove frameTime
        full_f = []
        full_name = []
        f_names = ['3_x', '3_y','4_x', '4_y', '6_x', '6_y','7_x','7_y']
        for f_name in f_names:
            try:
                signal = df[f_name]
                full_name.append(f_name+'_presence')
                full_f.append(1)
                full_name.append(f_name+'_pre_ratio')
                remove_na = signal.copy().dropna()
                full_f.append(float(len(remove_na))/float(len(df)))
            except:
                signal = pd.Series(0, index=np.arange(len(df)))
                full_name.append(f_name+'_presence')
                full_f.append(0)
                full_name.append(f_name+'_pre_ratio')
                full_f.append(0)
            
            full_name.append(f_name+'_mean')
            full_f.append(np.nanmean(signal))
            full_name.append(f_name+'_std')
            full_f.append(np.nanstd(signal))

        return [[full_name, full_f]]
    def apply_functionals(self,temp_signal):
        """
        we apply 18 functionals to the 1D signal to represent features over the whole video
        mean: mean value
        maxPos: the relative position of max value 
        minPos: the relative position of min value 
        stddev: standatd deviation
        kurtosis
        skewness
        quartile1: 25%
        quartile2: 50%
        quartile3: 75%
        iqr1_2: quartile2-quartile1
        iqr2_3: quartile3-quartile2
        iqr1_3: quartile3-quartile1
        percentile1:
        percentile99
        pcrtlrange0_1: percentile99.0-percentile1.0
        upleveltime75: time ratio for signal > (75%*range +min)
        upleveltime90: time ratio for signal > (90%*range +min)
        """
        name = ['mean','maxPos','minPos', 'stddev', 'kurtosis','skewness','quartile1','quartile2',
                'quartile3','iqr1_2','iqr2_3','iq1_3','percentile1','percentile99','pcrtlrange0_1','upleveltime50','upleveltime75','upleveltime90' ]
        
        mean = np.mean(temp_signal)
        maxPos = float(np.argmax(temp_signal)/len(temp_signal))
        minPos = float(np.argmin(temp_signal)/len(temp_signal))
        stddev = np.std(temp_signal)
        kurt = kurtosis(temp_signal)
        skewness = skew(temp_signal)
        quartile1 = np.percentile(temp_signal, 25)
        quartile2 = np.percentile(temp_signal, 50)
        quartile3 = np.percentile(temp_signal, 75)
        iqr1_2 = quartile2 - quartile1
        iqr2_3 = quartile3 - quartile2
        iqr1_3 = quartile3 - quartile1
        percentile1 = np.percentile(temp_signal,1)
        percentile99 = np.percentile(temp_signal, 99)
        pcrtlrange0_1 = percentile99 - percentile1
        upleveltime50 = float(sum(temp_signal> pcrtlrange0_1*0.5+percentile1) / len(temp_signal))
        upleveltime75 = float(sum(temp_signal>pcrtlrange0_1*0.75+percentile1) / len(temp_signal))
        upleveltime90 = float(sum(temp_signal>pcrtlrange0_1*0.9+percentile1) / len(temp_signal))
        
        functional_list = list([mean, maxPos, minPos, stddev, kurt, skewness, quartile1, quartile2, quartile3,
                                iqr1_2, iqr2_3, iqr1_3, percentile1, percentile99, pcrtlrange0_1,upleveltime50, upleveltime75, upleveltime90])
        
        return name, functional_list

    
    def calculate_derivatives(self,signal):
        # the input is :(x,y,z)
        derivatives = []
        for i in range(len(signal[0])):
            t_minus_1 = i
            t = (i+1)
            if t>=len(signal[0]):
                break
            x1,y1,z1 = signal[0][t_minus_1], signal[1][t_minus_1], signal[2][t_minus_1]
            x2,y2,z2 = signal[0][t], signal[1][t], signal[2][t]
            diff = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
            derivatives.append(diff)
        return np.asarray(derivatives)
    
    def calculate_magnitude(self, signal):
        #input: (x,y,z)
        magnitude = []
        for i in range(len(signal[0])):
            mag = np.sqrt(signal[0][i]**2+ signal[1][i]**2+ signal[2][i]**2)
            magnitude.append(mag)
            
        return np.asarray(magnitude)
    
    def functionals_for_au_presence(self, signal):
        # specially, for au presence, we only count the presence ratio, the frequency(per minue)
        pres_ratio = float(sum(signal==1)/ len(signal))
        # calculate transience
        diff = np.diff(signal)
        diff[diff==-1] = 0
        interval = int(0.5*self.rate)
        counter = 0
        for i in range(int(len(diff)//interval)):
            if (i+1)*interval>=len(diff):
                break
            window = diff[i*interval:(i+1)*interval]
            summation = sum(window)
            if summation>=1:
                counter+=1
            
        steps_per_minute = counter/float(len(signal)/float(self.rate*60))
        name = ['presence_ratio','freq_per_min']
        return name, [pres_ratio, steps_per_minute]


# hand crafter for audiof features
class Hand_Craftor_audio():
    def __init__(self):
        self.rate = 100 # 100Hz smaple rate
        self.metadata_path = '../labels_metadata.csv'
        self.test_meta_path = '../test_metadat.csv'
        self.mfcc_dir = '/newdisk/AVEC2018/downloaded_dataset/baseline_features/LLDs_audio_opensmile_MFCCs_turns'
        self.egmaps_dir = '/newdisk/AVEC2018/downloaded_dataset/baseline_features/features_audio_eGeMAPS_turns'
        self.emo_large_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/emo_large'
        self.emobase_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/emobase2010'
        self.VAD_dir = '/newdisk/AVEC2018/downloaded_dataset/VAD_turns/VAD_turns'
        self.seperator_dir = '/newdisk/AVEC2018/downloaded_dataset/sound_separator/sound_separator'
        self.transcript = pd.read_csv('Transcription.csv')
        self.audio_dir = '../recordings/recordings_audio'
        self.mfcc_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/mfcc.csv'
        self.egmaps_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/egmaps.csv'
        self.emo_large_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/emo_large.csv'
        self.emobase_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/emobase2010.csv'
        self.speech_path = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/speech.csv'
        self.feat_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/'
    def save_features(self):
        metadata = pd.read_csv(self.metadata_path, skipinitialspace=True)
        if args.test:
            metadata = pd.read_csv(self.test_meta_path,  skipinitialspace=True)
        if args.egmaps:
            print('Extracting for eGeMaps...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                egmaps_f = self.extract_eGeMAPS_features(instance_name)
                for f in egmaps_f:
                    f_name, f_value = f
                    for f_name_ins, f_value_ins in zip(f_name, f_value):
                        feature.loc[index, f_name_ins] = f_value_ins
            des = self.egmaps_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)
        if args.mfcc:
            print('Extracting for MFCC...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                mfcc_f = self.extract_mfcc_features(instance_name)
                for f in mfcc_f:
                    f_name, f_value = f
                    for f_name_ins, f_value_ins in zip(f_name, f_value):
                        feature.loc[index, f_name_ins] = f_value_ins
            des =self.mfcc_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)
        if args.emo_large:
            print('Extracting for Emolarge...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                file_path = os.path.join(self.emo_large_dir,instance_name+'.txt')
                data = list(self.read_outputfile(file_path))
                keys = ['emo_large_%s'%i for i in range(len(data))]
                for key, feat in zip(keys, data):
                    feature.loc[index, key] = feat
            des =self.emo_large_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)
        if args.emobase:
            print('Extracting for emobase2010...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                file_path = os.path.join(self.emobase_dir,instance_name+'.txt')
                data = list(self.read_outputfile(file_path))
                keys = ['emobase_%s'%i for i in range(len(data))]
                for key, feat in zip(keys, data):
                    feature.loc[index, key] = feat
            des =self.emobase_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)
        # the rest features needs to read audio, and transcription
        if args.word:
            print('Extracting for speech...')
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                instance_name = row['Instance_name']
                sample_rate, data = read(self.audio_dir+'/'+instance_name+'.wav')
                duration_second = len(data)/float(sample_rate)
                duration_min = duration_second/60
                utters = self.transcript.loc[self.transcript['video']==row['Instance_name']]
                transcript = ''
                for _,utter in utters.iterrows():
                    transcript = transcript+' '+utter['transcript']
                transcript = " ".join(transcript.split())
                words_num = len(transcript.split(' '))
                words_per_min = words_num/duration_min
                file_name = instance_name+'.csv'
                # the following features are about voice activity
                VAD_file = os.path.join(self.VAD_dir, file_name)
                df_v = pd.read_csv(VAD_file, header=None, sep=';')
                speech_ratio = np.sum(df_v[1]-df_v[0])/duration_second
                aver_speech_time = np.sum(df_v[1]-df_v[0])/len(df_v)
                max_speech_time = max(df_v[1]-df_v[0])
                sep_file = os.path.join(self.seperator_dir, file_name)
                df_s = pd.read_csv(sep_file, sep=';', header=None)
                aver_responce_time = np.mean(self.responce_time(df_s, df_v))
                max_responce_time = max(self.responce_time(df_s, df_v))
                feature.loc[index, 'words_num'] = words_num
                feature.loc[index,'voice_ratio'] = speech_ratio
                feature.loc[index,'aver_speech_time'] = aver_speech_time
                feature.loc[index,'max_speech_time'] = max_speech_time
                feature.loc[index,'aver_responce_time'] = aver_responce_time
                feature.loc[index,'max_responce_time'] = max_responce_time
                feature.loc[index, 'words_per_min'] = words_per_min
                feature.loc[index, 'duration_s'] = duration_second
                feature.loc[index, 'duration_m'] = duration_min
            des =self.speech_path
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            if not args.test:
                feature.to_csv(des, index=False)
            else:
                name = des.split('/')[-1].split('.')[0]
                name = name+'_test.csv'
                des = os.path.join(self.feat_dir, name)
                feature.to_csv(des, index=False)
        return feature
    def extract_eGeMAPS_features(self, instance_name):
        arff_file = os.path.join(self.egmaps_dir, instance_name+'.arff')
        data = arff.loadarff(arff_file)
        df = pd.DataFrame(data[0])
        df = df.loc[:, df.keys()[1:]] # remove frameTime
        features_all = []
        keys = df.keys()
        for key in keys:
            temp_signal = df.loc[:,key].values
            names, feature = self.apply_functionals(temp_signal)
            f_names = [key+'_'+n for n in names]
            features_all.append([f_names, feature])
        return features_all
    def extract_mfcc_features(self, instance_name):
        csv_file = os.path.join(self.mfcc_dir, instance_name+'.csv')
        df = pd.read_csv(csv_file, skipinitialspace=True, sep=';')
        df = df.loc[:, df.keys()[2:]] # remove frameTime and video name
        features_all = []
        keys = df.keys()
        for key in keys:
            temp_signal = df.loc[:,key].values
            names, feature = self.apply_functionals(temp_signal)
            f_names = [key+'_'+n for n in names]
            features_all.append([f_names, feature])
        return features_all
    def responce_time(self, df_sep, df_vad):
        start_times = df_sep[0]
        responce_times = []
        for start in start_times:
            try:
                speech_start = df_vad[0].loc[df_vad[0]>start].values[0]
                responce_times.append(speech_start-start)
            except:
                pass
        return responce_times
            
    def apply_functionals(self,temp_signal):
        name = ['mean','maxPos','minPos', 'stddev', 'kurtosis','skewness','quartile1','quartile2',
                'quartile3','iqr1_2','iqr2_3','iq1_3','percentile1','percentile99','pcrtlrange0_1','upleveltime50','upleveltime75','upleveltime90' ]
        
        mean = np.mean(temp_signal)
        maxPos = float(np.argmax(temp_signal)/len(temp_signal))
        minPos = float(np.argmin(temp_signal)/len(temp_signal))
        stddev = np.std(temp_signal)
        kurt = kurtosis(temp_signal)
        skewness = skew(temp_signal)
        quartile1 = np.percentile(temp_signal, 25)
        quartile2 = np.percentile(temp_signal, 50)
        quartile3 = np.percentile(temp_signal, 75)
        iqr1_2 = quartile2 - quartile1
        iqr2_3 = quartile3 - quartile2
        iqr1_3 = quartile3 - quartile1
        percentile1 = np.percentile(temp_signal,1)
        percentile99 = np.percentile(temp_signal, 99)
        pcrtlrange0_1 = percentile99 - percentile1
        upleveltime50 = float(sum(temp_signal> pcrtlrange0_1*0.5+percentile1) / len(temp_signal))
        upleveltime75 = float(sum(temp_signal>pcrtlrange0_1*0.75+percentile1) / len(temp_signal))
        upleveltime90 = float(sum(temp_signal>pcrtlrange0_1*0.9+percentile1) / len(temp_signal))
        
        functional_list = list([mean, maxPos, minPos, stddev, kurt, skewness, quartile1, quartile2, quartile3,
                                iqr1_2, iqr2_3, iqr1_3, percentile1, percentile99, pcrtlrange0_1,upleveltime50, upleveltime75, upleveltime90])
        
        return name, functional_list
    
    def clean_data(self, data):
        data = data.split(',')[1:] #'noname' deleted
        data = data[:-1] #'unknnwon deleted
        length = len(data)
        new_data = np.zeros(length)
        for i, item in enumerate(data):
            new_data[i] = float(item)
        return new_data
    def read_outputfile(self,outfile):
        file=open(outfile,'r')
        while True:
            line = file.readline()
            if line:
                if line.startswith('@data'):
                    line = file.readline()
                    line = file.readline()
                    data = line
                    if data: #sometimes , the data might be empty
                        data = self.clean_data(data)
                    break
            else:
                break
        return data     
  
import pdb
pdb.set_trace()
#hand_craftor = Hand_Craftor_visual()
#features = hand_craftor.save_features()
hand_craftor  = Hand_Craftor_audio()
va_features = hand_craftor.save_features()