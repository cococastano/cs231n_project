import extract_features
import data_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
            
## shuffle X and y in the same way
#m = len(y)
#rand_i = [i for i in range(m)]
#rand_i = np.random.permutation(np.array(rand_i))
#X = X[rand_i,:]
#y = y[rand_i,:]
#
## select k random example for starting features
#rand_i = np.random.randint(m, size=k)
#X_examples = X[rand_i,:]
#y_train = y[rand_i,:]
#
#thresh = 1e-8
#residual = 1
#iteration = 0
#
#while np.abs(residual) > thresh or iteration < 10:
#    X_examples_old = X_examples
#    dist = np.zeros((m,k))
#    # compute the euclidean distance
#    for i in range(k):
#        dist[:,i] = np.sqrt(np.sum(np.square(X_examples[i,:] - X), axis=1))
#    
#    # find closest feature for each example set
#    sorted_dist_i = np.argsort(dist, axis=1)
#    X_examples[sorted_dist_i[0]] = np.mean(X[sorted_dist_i[0],:], axis=0)
#    residual = np.linalg.norm(X_examples - X_examples_old)
#    iteration += 1
#    print('iteration: %d, residual: %f' % (iteration, residual))

# plot some stuff
plt.figure(0)  # LEO vs angle
plt.figure(1)  # LEO vs area
plt.figure(2)  # area vs angle
fig3D = plt.figure(3)  # LEO vs angle vs area
ax = fig3D.add_subplot(111, projection='3d')
for i in range(len(y)):
    LEO = X[i,0]
    area = X[i,1]
    angle = X[i,2]
    if y[i] == 1: color = 'r'
    if y[i] == 0: color = 'b'
    plt.figure(0)
    plt.plot(LEO,angle,color,marker='.',ms=10)
    plt.figure(1)
    plt.plot(LEO,area,color,marker='.',ms=10)
    plt.figure(2)
    plt.plot(area,angle,color,marker='.',ms=10)
    plt.figure(3)
    ax.scatter(LEO, angle, area, c=color)
plt.figure(0)
plt.show()
plt.xlabel('LEO')
plt.ylabel('leading droplet angle [rad]')
plt.figure(1)
plt.show()
plt.xlabel('LEO')
plt.ylabel('centroid polygon area')
plt.figure(2)
plt.show()
plt.xlabel('centroid polygon area')
plt.ylabel('leading droplet angle [rad]')
plt.figure(3)
plt.show()
ax.set_xlabel('LEO')
ax.set_ylabel('leading droplet angle [rad]')
ax.set_ylabel('centroid polygon area')


#while np.abs(residual) > thresh or iteration < 10:
#    X_train_old = X_train
#    dist = np.zeros((k,m))
#    # compute the euclidean distance
#    for i in range(k):
#        dist[:,i] = np.sqrt(np.sum(np.square(X_train[i,:] - X), axis=1))
#    
#    # find closest feature for each example set
#    my_sorted_label_indices = np.argsort(dist, axis=1)
#    y_pred = np.zeros(m)
#    for i in range(m):
#        closest_y = []
#        sorted_label_indices = my_sorted_label_indices[i,:]
#        closest_y = y_train[sorted_label_indices[:k]]
#        mode_label, count - stats.mode
#        y_pred[i] = np.argmax(np.bincount(closest_y))
    
#    i_min = numpy.where(dist == dist.min())
#    
#    i_min = np.argmin(dist,axis=1)
#    
#    ex_i = X[]
#    printnp.where(X==i_)
#    for i in range(k):
#        
#        X_train[i,:] = np.mean(X[np.where(X == i_min))

    