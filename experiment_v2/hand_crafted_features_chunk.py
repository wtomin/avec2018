#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:55:26 2018
extract hand crafted features based on chunks
@author: ddeng
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import skew, kurtosis
from VAD import VoiceActivityDetector
from scipy.io.wavfile import read
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-c','--chunk_way', type=str, default = 'equal',
                    help = 'which way to divide videos or audios: equal \
                    means chunk them every 10s, and bell means to be seperated by bell sound, none means do not chunk')

args = parser.parse_args()

class Hand_Craftor_visual():
    def __init__(self, args):
        self.rate = 30 # 30Hz smaple rate
        self.metadata_path = '../labels_metadata.csv'
        self.openface_dir = '../LLDs_video_openFace'
        self.sound_seperator = '../sound_seperator/sound_seperator'
        self.feature_csv = 'hand_crafted_features_chunk.csv'
        self.chunk_way = args.chunk_way
        self.clip_size = 10
    def get_features(self):
        if os.path.exists(self.feature_csv):
            feature = pd.read_csv(self.feature_csv)
        else:
            # in metadata, train_dev set are included
            metadata = pd.read_csv(self.metadata_path, skipinitialspace=True)
            col_names = ['Instance_name', 'Division', 'SubjectID', 'Age', 'Gender','Total_YMRS','ManiaLevel','Partition']
            feature = pd.DataFrame(columns=col_names) # empty dataframe
            div_index = 0
            for index, row in metadata.iterrows():
                file_name = row['Instance_name']+'.csv'
                dataframe = self.read_openface_csv(file_name)
                # clip dataframes
                if self.chunk_way=='bell':
                    bell_pos = pd.read_csv(os.path.join(self.sound_seperator,file_name),sep=';', header=None)
                    num_clip, dfs = self.split_dataframe(dataframe, bell_pos)
                else:
                    num_clip, dfs = self.split_dataframe(dataframe, self.clip_size)
                
                for div_name, df in enumerate(dfs):
                    feature.loc[div_index, 'Instance_name'] = row['Instance_name']
                    feature.loc[div_index, 'Dividion'] = div_name
                    feature.loc[div_index, 'SubjectID'] = row['SubjectID']
                    feature.loc[div_index, 'Age'] = row['Age']
                    feature.loc[div_index, 'Gender'] = row['Gender']
                    feature.loc[div_index, 'Total_YMRS'] = row['Total_YMRS']
                    feature.loc[div_index, 'ManiaLevel'] = row['ManiaLevel']
                    feature.loc[div_index, 'Partition'] = row['Partition']
                    #features about head motion
                    head_f = self.extract_head_features(df)
                    
                    for f in head_f:
                        f_name, f_value = f
                        for f_name_ins, f_value_ins in zip(f_name, f_value):
                            feature.loc[div_index, f_name_ins] = f_value_ins
                    
                    # features about gaze
                    gaze_f = self.extract_gaze_features(df)
                    for f in gaze_f:
                        f_name, f_value = f
                        for f_name_ins, f_value_ins in zip(f_name, f_value):
                            feature.loc[div_index, f_name_ins] = f_value_ins
                    
                    #feature about au
                    au_f = self.extract_au_features(df)
                    for f in au_f:
                        f_name, f_value = f
                        for f_name_ins, f_value_ins in zip(f_name, f_value):
                            feature.loc[div_index, f_name_ins] = f_value_ins
                    div_index+=1
            feature.to_csv(self.feature_csv, index=False)
        return feature
    def split_dataframe(self, dataframe, addition):
        #divide signal according to the chunk way
        if self.chunk_way=='equal':
            # 10s chunk
            time_duration = max(dataframe['timestamp'])
            time_clip = []
            for i in range(int(time_duration//addition)+1):
                if i != int(time_duration//addition):
                    time_clip.append([i*addition, (i+1)*addition])
                else:
                    if time_duration == int(time_duration//addition)*addition:
                        break
                    time_clip.append([i*addition, -1])
        elif self.chunk_way=='bell':
            time_clip = addition.values
        elif self.chunk_way=='none':
            time_clip=[0.0, time_duration]
        dfs = []
        for item in time_clip:
            start = item[0]
            if item[1]!=-1:
                end = item[1]
            else:
                end = time_duration
            df0 = dataframe.loc[dataframe['timestamp']>=start]
            df1 = df0.loc[df0['timestamp']<end]
            df1.reset_index(drop =True, inplace=True)
            assert len(df1)!=0
            dfs.append(df1)
        return len(dfs), dfs
    def read_openface_csv(self, file_name):

        path = os.path.join(self.openface_dir, file_name)
    
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
        x,y = dataframe['gaze_angle_x'], dataframe['gaze_angle_y']
        signal = [x,y]
        
        f_name_x, f_x= self.apply_functionals(signal[0])
        f_name_x = ['gaze_x_'+name for name in f_name_x]

        f_name_y, f_y = self.apply_functionals(signal[1])
        f_name_y = ['gaze_y_'+name for name in f_name_y]

        return [[f_name_x, f_x],[f_name_y, f_y]]
    
    def extract_au_features(self,dataframe):
        full_name = []
        full_f = []
        # intensity(0,5)
        f_name_cate = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r','AU07_r', 
                        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
        for f_name in f_name_cate:
            signal = dataframe[f_name]
            f_name_inten, f_inten = self.apply_functionals(signal)
            f_name_inten = [f_name+'_'+name for name in f_name_inten]
            full_name.extend(f_name_inten)
            full_f.extend(f_inten)
        
        f_name_cate =[ 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 
                      'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 
                      'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
        for f_name in f_name_cate:
            signal = dataframe[f_name]
            f_name_pres, f_pres = self.functionals_for_au_presence(signal)
            f_name_pres = [f_name+'_'+name for name in f_name_pres]
            full_name.extend(f_name_pres)
            full_f.extend(f_pres)
            
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
            
        steps_per_minute = float(counter/(len(signal)/(self.rate*60)))
        name = ['presence_ratio','freq_per_min']
        return name, [pres_ratio, steps_per_minute]


# hand crafter for audiof features
class Hand_Craftor_audio():
    def __init__(self, inherit_csv = None):
        self.rate = 100 # 100Hz smaple rate
        self.metadata_path = '../labels_metadata.csv'
        self.openface_dir = '../LLDs_audio_opensmile'
        self.audio_feature_dir0 = '../LLDs_audio_opensmile/LLDs_audio_opensmile_MFCCs'
        self.audio_feature_dir1 = '../LLDs_audio_opensmile/LLDs_audio_eGeMAPS'
        self.feature_csv = 'hand_crafted_features_chunk_va.csv'
        self.inherit_csv = inherit_csv
        self.transcript = pd.read_csv('Transcription.csv')
        self.audio_dir = '../recordings/recordings_audio'
        self.chunk_way = args.chunk_way
        self.clip_size = 10
        self.sound_seperator = '../sound_seperator/sound_seperator'
    def get_features(self):
        if os.path.exists(self.feature_csv):
            feature = pd.read_csv(self.feature_csv)
        else:
            feature =self.inherit_csv.copy()
            for index, row in feature.iterrows():
                # these features need to read opensmile csv
                file_name = row['Instance_name']+'.csv'
                division = row['Division']
                if self.chunk_way=='bell':
                    bell_pos = pd.read_csv(os.path.join(self.sound_seperator,file_name),sep=';', header=None)
                dataframe = self.read_opensmile_csv(file_name)
                if self.chunk_way=='bell':
                    splitted_frame = self.get_splitted_dataframe(dataframe, division, bell_pos)
                else:
                    splitted_frame = self.get_splitted_dataframe(dataframe, division, None)
                f_name, f_value = self.extract_audio_features(splitted_frame)
                for f_name_ins, f_value_ins in zip(f_name, f_value):
                    feature.loc[index, f_name_ins] = f_value_ins
                
#                # the rest features needs to read audio, and transcription
#                sample_rate, data = read(self.audio_dir+'/'+row['Instance_name']+'.wav')
#                duration_second = len(data)*(1.0/sample_rate)
#                duration_min = duration_second/60
#                utters = self.transcript.loc[self.transcript['video']==row['Instance_name']]
#                transcript = ''
#                for _,utter in utters.iterrows():
#                    transcript = transcript+' '+utter['transcript']
#                transcript = " ".join(transcript.split())
#                words_num = len(transcript.split(' '))
#                words_per_min = words_num/duration_min
#                
#                # the following features are about voice activity
#                v = VoiceActivityDetector(sample_rate, data)
#                speech_ratio = v.detect_speech()
#                feature.loc[index, 'words_num'] = words_num
#                feature.loc[index,'voice_ratio'] = speech_ratio
#                feature.loc[index, 'words_per_min'] = words_per_min
#                feature.loc[index, 'duration_s'] = duration_second
#                feature.loc[index, 'duration_m'] = duration_min
                
            feature.to_csv(self.feature_csv, index=False)
        return feature
     
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
    def get_splitted_dataframe(self, dataframe, div_index, bell_pos):
        #divide signal according to the chunk way
        if self.chunk_way=='equal':
            # 10s chunk
            time_duration = max(dataframe['frameTime'])
            if div_index!= int(time_duration//self.clip_size):
                start = div_index*self.clip_size
                end = (div_index+1)*self.clip_size
            else:
                start = div_index*self.clip_size
                end = time_duration
            time_clip = [start, end]
        elif self.chunk_way=='bell':
            time_clip = np.squeeze(bell_pos.values[div_index])
        elif self.chunk_way=='none':
            time_clip=[0.0, time_duration]

        start = time_clip[0]
        if time_clip[1]!=-1:
            end = time_clip[1]
        else:
            end = time_duration
        df0 = dataframe.loc[dataframe['frameTime']>=start]
        df1 = df0.loc[df0['frameTime']<end]
        df1.reset_index(drop =True, inplace=True)
        assert len(df1)!=0

        return df1
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
            
    def cut_front_and_tail(self, signal):
        # delete the first and last 2s
        
        interval = 2 * self.rate
        chopped_signal=[]
        for i in range(len(signal)):
            crop_s = signal[i][interval:-interval]
            crop_s.reset_index(drop =True, inplace=True)
            chopped_signal.append(crop_s)
        
        return chopped_signal    
hand_craftor = Hand_Craftor_visual(args)
features = hand_craftor.get_features()
hand_craftor  = Hand_Craftor_audio(features)
va_features = hand_craftor.get_features()
