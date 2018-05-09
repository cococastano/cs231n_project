import cv2
import numpy as np
import os
import random
import data_utils

def get_droplets_centroids(frame):
    """
    Input a frame and return list of tuples with x and y centroid coordinates
    """

    M = cv2.moments(frame)
    
#    cx = int(M['m10']/M['m00'])
#    cy = int(M['m01']/M['m00'])
    
#    manip_frame = 
#    frame = cv2.bitwise_not(frame)
    orig_frame = frame
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    _, my_contours, _ = cv2.findContours(orig_frame, cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_NONE) 
    centroids = [0] * len(my_contours)
    for i, cont in enumerate(my_contours):
        M = cv2.moments(cont)
        print(M['m00'],M['m01'],M['m10'])
        if M['m00'] != 0: 
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX = 0
            cY = 0
        centroids[i] = (cX, cY)
        cv2.circle(rgb_frame, (cX,cY), 1, (0,0,255), 7)
        cv2.putText(rgb_frame, str(i), (cX - 4, cY - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow('centroids', rgb_frame)
    
    return centroids
    

frames = data_utils.pull_frame_range()
test_frame = frames['break0'][0]
#data_utils.show_countours(test_frame,resize_frame=3)

centroids = get_droplets_centroids(test_frame)


# cv2.imshow('image',frames['break0'][0])