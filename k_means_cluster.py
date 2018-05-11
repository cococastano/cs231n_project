import extract_features
import data_utils
import cv2
import numpy as np
import os
import random

# number of clusters
k = 5
n_features = 3

# load data
my_frames = data_utils.pull_frame_range(frame_range=[3])
# construct X matrix and y vector
X = np.zeros((len(my_frames), n_features))
y = np.zeros((len(my_frames), 1))
for i, key in enumerate(my_frames):
    frame = my_frames[key][0]
    LEO = key.split('LEO_')[-1]
    centroids = extract_features.get_n_leading_droplets_centroids(frame, n=3)
    area = extract_features.polygon_area(centroids=centroids)
    leading_angle = extract_features.leading_angle(centroids=centroids)
    
    X[i,0] = LEO
    X[i,1] = area
    X[i,2] = leading_angle
    
    # classify a break as 0 and nobreak as 1
    my_class = key.split('_')[0]
    
    if 'nobreak' in my_class:
        y[i] = 1
    else:
        y[i] = 0
            
# shuffle X and y in the same way
m = len(y)
rand_i = [i for i in range(m)]
rand_i = np.random.permutation(np.array(rand_i))
X = X[rand_i,:]
y = y[rand_i,:]

# select k random example for starting features
rand_i = np.random.randint(m, size=k)
mu = X[rand_i,:]

thresh = 1e-8
residual = 1
iteration = 0

while np.abs(residual) > thresh or iteration < 10:
    mu_old = mu
    dist = np.zeros((m,k))
    # compute the euclidean distance
    for i in range(k):
        dist[:,i] = np.sqrt(np.sum(np.square(mu[i,:] - X), axis=1))
    
    # find closest feature for each example set
    i_min = np.argmin(dist,axis=1)

