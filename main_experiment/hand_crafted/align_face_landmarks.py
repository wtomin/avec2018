#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:23:01 2018

@author: ddeng
"""
from scipy.spatial import procrustes as pc
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import cv2

# Open images...
target_X_img = cv2.imread('/newdisk/AVEC2018/downloaded_dataset/openface_faces/dev_001/dev_001_aligned/frame_det_00_000001.bmp',0)
input_Y_img = cv2.imread('/newdisk/AVEC2018/downloaded_dataset/openface_faces/dev_001/dev_001_aligned/frame_det_00_000148.bmp',0)

# Landmark points - same number and order!
# l eye, r eye, nose tip, l mouth, r mouth
X_pts = np.asarray([[61,61],[142,62],[101,104],[71,143],[140,139]])
Y_pts = np.asarray([[106,91],[147,95],[129,111],[104,130],[141,135]])

# Calculate transform via procrustes...
d,Z_pts,Tform = pc(X_pts,Y_pts)

# Build and apply transform matrix...
# Note: for affine need 2x3 (a,b,c,d,e,f) form
R = np.eye(3)
R[0:2,0:2] = Tform['rotation']
S = np.eye(3) * Tform['scale'] 
S[2,2] = 1
t = np.eye(3)
t[0:2,2] = Tform['translation']
M = np.dot(np.dot(R,S),t.T).T
tr_Y_img = cv2.warpAffine(input_Y_img,M[0:2,:],(400,400))

# Confirm points...
aY_pts = np.hstack((Y_pts,np.array(([[1,1,1,1,1]])).T))
tr_Y_pts = np.dot(M,aY_pts.T).T

# Show result - input transformed and superimposed on target...
plt.figure() 
plt.subplot(1,3,1)
plt.imshow(target_X_img,cmap=cm.gray)
plt.plot(X_pts[:,0],X_pts[:,1],'bo',markersize=5)
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(input_Y_img,cmap=cm.gray)
plt.plot(Y_pts[:,0],Y_pts[:,1],'ro',markersize=5)
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(target_X_img,cmap=cm.gray)
plt.imshow(tr_Y_img,alpha=0.6,cmap=cm.gray)
plt.plot(X_pts[:,0],X_pts[:,1],'bo',markersize=5) 
plt.plot(Z_pts[:,0],Z_pts[:,1],'ro',markersize=5) # same as...
plt.plot(tr_Y_pts[:,0],tr_Y_pts[:,1],'gx',markersize=5)
plt.axis('off')
plt.show()