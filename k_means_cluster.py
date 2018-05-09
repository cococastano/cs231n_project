import extract_features
import data_utils
import cv2
import numpy as np
import os
import random

frames = data_utils.pull_frame_range()
print(np.shape(frames['break0'][0]))
test_frame = frames['break0'][0]
moments = extract_features.get_droplets_centroids(test_frame)
cv2.imshow('image',frames['break0'][0])