#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:00:40 2018

@author: ddeng
"""

import os
import numpy as np
import glob
def clean_data(data):
    data = data.split(',')[1:] #'noname' deleted
    data = data[:-1] #'unknnwon deleted
    length = len(data)
    new_data = np.zeros(length)
    for i, item in enumerate(data):
        new_data[i] = float(item)
    return new_data
def read_outputfile(outfile):
    file=open(outfile,'r')
    while True:
        line = file.readline()
        if line:
            if line.startswith('@data'):
                line = file.readline()
                line = file.readline()
                data = line
                if data: #sometimes , the data might be empty
                    data = clean_data(data)
                break
        else:
            break
    return data
extractor = '/home/ddeng/opensmile-2.3.0/SMILExtract'
indir = '/newdisk/AVEC2018/downloaded_dataset/recordings/recordings_audio'
outdir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/emobase2010'
if not os.path.exists(outdir):
    os.makedirs(outdir)
config='/home/ddeng/opensmile-2.3.0/config/emobase2010.conf'
audios = glob.glob(os.path.join(indir, '*.wav'))

for audio in audios:
    instance_name = audio.split('/')[-1].split('.')[0]
    outfile = os.path.join(outdir, instance_name+'.txt')
    infile = audio
    cmd = extractor+' -C '+ config +' -I '+infile+' -O '+outfile
    os.system(cmd)