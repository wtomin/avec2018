#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:19:50 2018
extract hand crafted features: head motion, gaze motion, AUs, audio low level features, voice ratio, average speaking rate
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
import argparse
parser = argparse.ArgumentParser()  
parser.add_argument("--head", action="store_false",help="whether to add head features")  
parser.add_argument("--au" ,action="store_false", help="whether to add au features")  
parser.add_argument("--gaze", action="store_false" ,help="whether to add gaze features")
parser.add_argument("--pose", action="store_false", help="whether to add pose features")
parser.add_argument("--lld_audio", action="store_false", help="whether to add lld audio features")
parser.add_argument("--word", action="store_false", help="whether to add word features")
parser.add_argument("--overwrite", action="store_false", help="whether to overwrite features")
args = parser.parse_args()  

class Hand_Craftor_visual():
    def __init__(self):
        self.rate = 30 # 30Hz smaple rate
        self.metadata_path = '../labels_metadata.csv'
        self.openface_dir = '/newdisk/AVEC2018/downloaded_dataset/openface_faces'
        self.gaze_file_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/gaze_angle'
        self.pose_dir = '../recordings/video_pose'
        self.feature_csv = 'hand_crafted_features.csv'

    def get_features(self):
        if os.path.exists(self.feature_csv) and not args.overwrite:
            feature = pd.read_csv(self.feature_csv)
        else:
            # in metadata, train_dev set are included
            metadata = pd.read_csv(self.metadata_path, skipinitialspace=True)
            feature = metadata.copy()
            for index, row in tqdm(metadata.iterrows()):
                pass
                file_name = row['Instance_name']+'.csv'
                dataframe = self.read_openface_csv(file_name)
                #features about head motion
                if args.head:
                    head_f = self.extract_head_features(dataframe)
                    for f in head_f:
                        f_name, f_value = f
                        for f_name_ins, f_value_ins in zip(f_name, f_value):
                            feature.loc[index, f_name_ins] = f_value_ins
                
                #feature about au
                if args.au:
                    au_f = self.extract_au_features(dataframe)
                    for f in au_f:
                        f_name, f_value = f
                        for f_name_ins, f_value_ins in zip(f_name, f_value):
                            feature.loc[index, f_name_ins] = f_value_ins
                
                # features about gaze, average fixation time, angle difference
                if args.gaze:
                    dataframe2 = pd.read_csv(os.path.join(self.gaze_file_dir, file_name), skipinitialspace=True)
                    gaze_f = self.extract_gaze_features(dataframe2)
                    for f in gaze_f:
                        f_name, f_value = f
                        for f_name_ins, f_value_ins in zip(f_name, f_value):
                            feature.loc[index, f_name_ins] = f_value_ins
                # then add pose features
                if args.pose:
                    pose_path = os.path.join(self.pose_dir, file_name)
                    dataframe3 = pd.read_csv(pose_path, skipinitialspace=True)
                    
                    pose_f = self.extract_pose_features(dataframe3)
                    
                    f_name, f_value = pose_f
                    for f_name_ins, f_value_ins in zip(f_name, f_value):
                        feature.loc[index, f_name_ins] = f_value_ins
            print('Whether to contain NaN value:{}'.format(feature.isnull().sum().sum()))
            feature.to_csv(self.feature_csv, index=False)
        
        return feature
    def read_openface_csv(self, file_name):

        path = os.path.join(self.openface_dir, file_name.split('.')[0], file_name)
    
        dataframe = pd.read_csv(path, skipinitialspace=True)
        return dataframe
    def extract_head_features(self, dataframe):
        x,y,z = dataframe['pose_Tx'], dataframe['pose_Ty'], dataframe['pose_Tz']
        signal = [x,y,z]
        
        # derivatives
        derivatives = self.calculate_derivatives(signal)
        f_name_der, f_der = self.apply_functionals(derivatives)
        f_name_der = ['head_der_'+name for name in f_name_der]
        
        # magnitude
        magnitudes = self.calculate_magnitude(signal)
        f_name_mag, f_mag = self.apply_functionals(magnitudes)
        f_name_mag = ['head_mag_'+name for name in f_name_mag]
        
        return [[f_name_der, f_der], [f_name_mag, f_mag]]
    def extract_gaze_features(self,dataframe):
        angle_velocities = dataframe['angle_velocity']
        fixation = dataframe['fixation']
        # angle related features
        f_name_angle, f_angle  = self.apply_functionals(angle_velocities)
        f_name_angle = ['angle_'+name for name in f_name_angle]
        
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
        return [[f_name_angle, f_angle], [f_name_fix, f_fix]]
    def extract_au_features(self,dataframe):
        full_name = []
        full_f = []
        # intensity(0,5)
        f_name_cate = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r','AU07_r', 
                        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
        for f_name in f_name_cate:
            f_name_inten, f_inten = self.apply_functionals(dataframe[f_name])
            f_name_inten = [f_name+'_'+name for name in f_name_inten]
            full_name.extend(f_name_inten)
            full_f.extend(f_inten)
        
        f_name_cate =[ 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 
                      'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 
                      'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
        for f_name in f_name_cate:
            f_name_pres, f_pres = self.functionals_for_au_presence(dataframe[f_name])
            f_name_pres = [f_name+'_'+name for name in f_name_pres]
            full_name.extend(f_name_pres)
            full_f.extend(f_pres)
            
        return [[full_name, full_f]]
    def extract_pose_features(self, dataframe):
        # except 3,4,6,7, other joints only calculate it's std
        # for 3,4,6,7, if not appear, zero; if appear, presence 1, and mean value for x and y
        full_name = []
        full_f = []
        f_names = []
        for i in range(18):
            if i in [8,9,10,11,12,13]:
                pass
            else:
                f_names.append(str(i)+'_x')
                f_names.append(str(i)+'_y')
        for f_name in f_names:
            if f_name not in ['3_x', '3_y','4_x', '4_y', '6_x', '6_y','7_x','7_y']:
                try:
                    signal = dataframe[f_name]
                except:
                    signal = pd.Series(0, index=np.arange(len(dataframe)))
                full_name.append(f_name+'_mean')
                full_f.append(np.nanmean(signal))
                full_name.append(f_name+'_std')
                full_f.append(np.nanstd(signal))
                
#                derivatives = np.gradient(signal.values)
#                full_name.extend(f_name+'_der_mean')
#                full_f.extend(np.mean(derivatives))
#                full_name.extend(f_name+'_der_std')
#                full_f.extend(np.std(derivatives))
            else:
                try:
                    signal = dataframe[f_name]
                    full_name.append(f_name+'_presence')
                    full_f.append(1)
                    full_name.append(f_name+'_mean')
                    full_f.append(np.nanmean(signal))
                    full_name.append(f_name+'_std')
                    full_f.append(np.nanstd(signal))
                except:
                    full_name.append(f_name+'_presence')
                    full_f.append(0)
                    full_name.append(f_name+'_mean')
                    full_f.append(0)
                    full_name.append(f_name+'_std')
                    full_f.append(0)
                    
        return [full_name, full_f]
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
    def __init__(self, inherit_csv = None):
        self.rate = 100 # 100Hz smaple rate
        self.metadata_path = '../labels_metadata.csv'
        self.audio_feature_dir0 = '../LLDs_audio_opensmile/LLDs_audio_opensmile_MFCCs'
        self.audio_feature_dir1 = '../LLDs_audio_opensmile/LLDs_audio_eGeMAPS'
        self.feature_csv = 'hand_crafted_features_va.csv'
        self.inherit_csv = inherit_csv
        self.transcript = pd.read_csv('Transcription.csv')
        self.audio_dir = '../recordings/recordings_audio'
        self.VAD_dir = '/newdisk/AVEC2018/downloaded_dataset/VAD_turns/VAD_turns'
        self.seperator_dir = '/newdisk/AVEC2018/downloaded_dataset/sound_separator/sound_separator'
    def get_features(self):
        if os.path.exists(self.feature_csv) and not args.overwrite:
            feature = pd.read_csv(self.feature_csv)
        else:
            if self.inherit_csv is None:
                self.inherit_csv = pd.read_csv(self.metadata_path, skipinitialspace=True) 
            feature =self.inherit_csv.copy()
            for index, row in tqdm(feature.iterrows()):
                pass
                # these features need to read opensmile csv
                file_name = row['Instance_name']+'.csv'
                dataframe = self.read_opensmile_csv(file_name)
                if args.lld_audio:
                    f_name, f_value = self.extract_audio_features(dataframe)
                    for f_name_ins, f_value_ins in zip(f_name, f_value):
                        feature.loc[index, f_name_ins] = f_value_ins
                
                # the rest features needs to read audio, and transcription
                if args.word:
    #                sample_rate, data = read(self.audio_dir+'/'+row['Instance_name']+'.wav')
                    duration_second = max(dataframe['frameTime'])
                    duration_min = duration_second/60
                    utters = self.transcript.loc[self.transcript['video']==row['Instance_name']]
                    transcript = ''
                    for _,utter in utters.iterrows():
                        transcript = transcript+' '+utter['transcript']
                    transcript = " ".join(transcript.split())
                    words_num = len(transcript.split(' '))
                    words_per_min = words_num/duration_min
                    
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
                
            feature.to_csv(self.feature_csv, index=False)
        return feature
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
            
    def read_opensmile_csv(self, file_name):

        path0 = os.path.join(self.audio_feature_dir0, file_name)
        path1 = os.path.join(self.audio_feature_dir1, file_name)
        
        dataframe0 = pd.read_csv(path0, skipinitialspace=True, delimiter=';')
        dataframe1 = pd.read_csv(path1,skipinitialspace=True, delimiter=';')
    
        merge_df = pd.concat([dataframe0, dataframe1], axis =1)
        merge_df = merge_df.loc[:,~merge_df.columns.duplicated()]
        merge_df = merge_df.dropna()
        return merge_df
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
    def extract_audio_features(self, dataframe):
        feature_categories = list(dataframe.keys()[2:])
        full_f_name = []
        full_f= []
        for f_c in feature_categories:
            signal = dataframe[f_c]
            f_name, f_value = self.apply_functionals(signal)
            f_name = [f_c+'_'+name for name in f_name]
            full_f_name.extend(f_name)
            full_f.extend(f_value)
        return full_f_name, full_f
            
  
#import pdb
#pdb.set_trace()
hand_craftor = Hand_Craftor_visual()
features = hand_craftor.get_features()
hand_craftor  = Hand_Craftor_audio(features)
va_features = hand_craftor.get_features()