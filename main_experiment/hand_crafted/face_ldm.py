#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:40:38 2018
face landmarks motion
@author: ddeng
"""

import numpy
import pandas as pd
import os
import os.path as path
from tqdm import tqdm
import matplotlib.pyplot as plt

refer_index = [33, 36,39,42,45] # left eye corner, right eye corner, nose
LEFT_EYEBROW = list(numpy.arange(17, 22))
RIGHT_EYEBROW = list(numpy.arange(22, 27))
LEFT_EYE = [37, 38, 40,41]
RIGHT_EYE = [43, 44, 46, 47]
MOUTH_CENTER = [50,51,52,56,57,58,61,62,63, 65, 66, 67]
MOUTH_CORNER = [48, 54,60, 64 ]
GROUPS = [LEFT_EYE, LEFT_EYEBROW, RIGHT_EYE, RIGHT_EYEBROW, MOUTH_CENTER, MOUTH_CORNER]
GROUPS_NAMES = ['LEFT_EYE', 'LEFT_EYEBROW', 'RIGHT_EYE', 'RIGHT_EYEBROW', 'MOUTH_CENTER', 'MOUTH_CORNER']
def display_aligned_face(aligned_landmarks):
    # display three 
    plt.figure()
    img_0=aligned_landmarks[0]
    X = [item[0] for item in img_0]
    Y = [item[1] for item in img_0]
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    A = plt.scatter(X,Y, marker='o')
    img_1 = aligned_landmarks[30]
    X = [item[0] for item in img_1]
    Y = [item[1] for item in img_1]
    B = plt.scatter(X, Y,marker='x')
    img_2 = aligned_landmarks[100]
    X = [item[0] for item in img_2]
    Y = [item[1] for item in img_2]
    C = plt.scatter(X, Y,marker='*')
    plt.legend((A, B,C),('Fram0', 'Frame30', 'Frame100'), loc='lower left')
    plt.axis('off')
    plt.show()
def display_one_face(aligned_landmarks):
    # display three 
    plt.figure()
    img_0=aligned_landmarks[0]
    X = [item[0] for item in img_0]
    Y = [item[1] for item in img_0]
    text = [str(i) for i in range(68)]
    for i, txt in enumerate(text):
        plt.annotate(txt, (X[i]-4, Y[i]-2), fontsize=13)
#    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    LEFT_EYE_EX = LEFT_EYE.copy()
    LEFT_EYE_EX.extend([36, 39])
    RIGHT_EYE_EX = RIGHT_EYE.copy()
    RIGHT_EYE_EX.extend([42, 45])
    for index in LEFT_EYE_EX:
        plt.scatter(X[index], Y[index], color='r', marker ='x')
    for index in LEFT_EYEBROW:
        plt.scatter(X[index], Y[index], color='r', marker ='x')
    for index in RIGHT_EYE_EX:
        plt.scatter(X[index], Y[index], color='r', marker ='x')
    for index in RIGHT_EYEBROW:
        plt.scatter(X[index], Y[index], color='r', marker ='x')
    for index in MOUTH_CENTER:
        plt.scatter(X[index], Y[index], color='r', marker ='x')
    for index in MOUTH_CORNER:
        plt.scatter(X[index], Y[index], color='r', marker ='x')
        
    ALL_LMK=LEFT_EYE_EX.copy()
    ALL_LMK.extend(LEFT_EYEBROW)
    ALL_LMK.extend(RIGHT_EYE_EX)
    ALL_LMK.extend(RIGHT_EYEBROW)
    ALL_LMK.extend(MOUTH_CENTER)
    ALL_LMK.extend(MOUTH_CORNER)
    X = [X[i] for i in range(68) if i not in ALL_LMK ]
    Y = [Y[i] for i in range(68) if i not in ALL_LMK ]
    plt.scatter(X,Y, marker='o', color='b')


    plt.axis('off')

    plt.show()

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
    
        sum || s*R*p1,i + T - p2,i||^2
        
    is minimized.
    """

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)


    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])
def get_affine_matrix(orgi_landmarks, tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    return M

def get_face_aligned_landmarks(csv_file):
    df = pd.read_csv(csv_file, skipinitialspace=True)
    df = df.loc[df['success']==1]
    df.reset_index(drop=True, inplace=True)
    x_keys = ['x_%s'%str(i) for i in range(68)]
    y_keys = ['y_%s'%str(i) for i in range(68)]
    aligned_landmarks = []
    for index, row in df.iterrows():
        face_landmark = []
        for x, y in zip(x_keys,y_keys):
            x_coor = row[x]
            y_coor = row[y]
            face_landmark.append([x_coor, y_coor])
        if index==0:
            refer_landmark = [face_landmark[idx] for idx in refer_index] #left and right eye corner, nose
            aligned_face_landmark=face_landmark
        else:
            # affine
            orgi_ldm = [face_landmark[idx] for idx in refer_index]
            M = get_affine_matrix(orgi_ldm, refer_landmark)
            aligned_face_landmark = []
            for item in face_landmark:
                ldm = item.copy()
                ldm.append(1)
                aligned_ldm = numpy.dot(M, numpy.matrix(ldm).T)[:2]
                aligned_face_landmark.append(aligned_ldm)

        aligned_landmarks.append(aligned_face_landmark)
    display_one_face(aligned_landmarks)    
    length = len(aligned_landmarks)
    for index, frame_ldm in enumerate(aligned_landmarks):
        if index==length-1:
            break
        current_frame = frame_ldm
        next_frame = aligned_landmarks[index+1]
        group_motion = get_group_motion(current_frame, next_frame)
        distances = geometric_distance(current_frame)
        merged_dict = group_motion.copy()
        merged_dict.update(distances)
        dataframe = pd.DataFrame.from_dict(merged_dict)
        if index==0:
            features = dataframe
        else:
            features = features.append(dataframe, ignore_index=True)
    return features

def get_group_motion(curr_frame_ldm, next_frame_ldm):
    group_motion = {}
    #
    for i, group in enumerate(GROUPS):
        curr_groups = [curr_frame_ldm[index] for index in group]
        next_groups  = [next_frame_ldm[index] for index in group]
        largest_disp = 0
        for curr_coor, next_coor in zip(curr_groups, next_groups):
            x0, y0 = list(curr_coor)
            x1, y1 = list(next_coor)
            x0, x1, y0,y1 = float(x0), float(x1), float(y0), float(y1)
            displace = numpy.sqrt((x1-x0)**2+(y1-y0)**2)
            if displace>largest_disp:
                largest_disp=displace
        group_motion[GROUPS_NAMES[i]] = [largest_disp]
    return group_motion
def geometric_distance(landmarks):
    distances = {}
    #eye open size
    upper = [37, 38]
    down = [41, 40]
    distances['eye_open_l'] = calculate_distance(upper, down, landmarks)
    upper = [43, 44]
    down = [47, 46]
    distances['eye_open_r'] = calculate_distance(upper, down, landmarks)
    # eye_to_eyebrow
    upper = [17, 18, 19, 20]
    down = [36, 37, 38, 39]
    distances['eye_to_eyebrow_l'] = calculate_distance(upper, down, landmarks)
    upper = [23, 24, 25, 26]
    down = [42, 43, 44, 45]
    distances['eye_to_eyebrow_r'] = calculate_distance(upper, down, landmarks)
    #mouth
    upper = [50, 51, 52]
    down = [58, 57, 56]
    distances['mouth_open'] = calculate_distance(upper, down, landmarks)
    upper = [48]
    down = [54]
    distances['mouth_corner'] = calculate_distance(upper, down, landmarks)
    return distances
def calculate_distance(upper, down, landmarks):

    upper_lmd = [landmarks[index] for index in upper]
    down_lmd = [landmarks[index] for index in down]
    mean_distance = 0
    for upside, downside in zip(upper_lmd, down_lmd):
        x0,y0 = upside
        x1,y1 = downside
        x0, x1, y0,y1 = float(x0), float(x1), float(y0), float(y1)
        distance = numpy.sqrt((x1-x0)**2+(y1-y0)**2)
        mean_distance+=distance
    mean_distance = mean_distance/len(upper)
    return [mean_distance]
openface_dir = '/newdisk/AVEC2018/downloaded_dataset/openface_faces'
des_dir = '/newdisk/AVEC2018/downloaded_dataset/hand_crafted_features/face'
if not path.exists(des_dir):
    os.makedirs(des_dir)
    
videos =  os.listdir(openface_dir)
for video_name in tqdm(videos):
    pass
    if path.isdir(path.join(openface_dir, video_name)):
        csv_file_path = path.join(openface_dir, video_name, video_name+'.csv')
        features = get_face_aligned_landmarks(csv_file_path)
        des = path.join(des_dir,video_name+'.csv')
        features.to_csv(des, index=False)
        
