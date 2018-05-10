import cv2
import numpy as np
import os
import random
import data_utils
import matplotlib.pyplot as plt

def get_droplets_info(frame):
    """
    Input: a frame
    Return:
         A list of tuples with x and y centroid coordinates
         A list of areas of drops
         A list of tuples with contours of drops
    """
    orig_frame = frame
    # rgb_frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    _, my_contours, _ = cv2.findContours(orig_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centroids = []
    areas = []
    contours = []
    for contour in my_contours:
        # remove the contour of channel and the gap between drops
        if (len(contour) < 300) & (len(contour) > 10):
            areas.append(cv2.contourArea(contour))
            contours.append(contour)
            M = cv2.moments(contour)
            # print(M['m00'],M['m01'],M['m10'])
            if M['m00'] != 0:
                centroid = [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]
            else:
                centroid = [0, 0]
            centroids.append(centroid)
            # plt.plot(contour[:, :, 0], contour[:, :, 1])
        
    return centroids, areas, contours
    

frames = data_utils.pull_frame_range()
test_frame = frames['break0'][0]
# cont_frame = data_utils.show_countours(test_frame,resize_frame=3)

centroids, areas, contours = get_droplets_info(test_frame)

# get more features from frame
# for c in centroids:
    

# cv2.imshow('image',frames['break0'][0])