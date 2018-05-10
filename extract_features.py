import cv2
import numpy as np
import os
import random
import data_utils

def get_droplets_centroids(frame):
    """
    Input a frame and return list of tuples with x and y centroid coordinates
    sorted by x coordinate (left to right) and with non-droplet regions
    removed
    """
    orig_frame = frame
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    # get contours
    _, my_contours, _ = cv2.findContours(orig_frame, cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_NONE)
    # remove the two largest contours which are not drops
    my_contours = sorted(my_contours, key=cv2.contourArea, 
                         reverse=True)[2:]
    # find centroids from the moments
    centroids = [0] * len(my_contours)
    for i, cont in enumerate(my_contours):
        M = cv2.moments(cont)
        if M['m00'] != 0: 
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX = 0
            cY = 0
        centroids[i] = (cX, cY)
    # sort list by first element in tuple
    centroids.sort(key=lambda x: x[0], reverse=True)
        
    return centroids
    

frames = data_utils.pull_frame_range()
test_frame = frames['break0'][0]

cont_frame = data_utils.show_my_countours(test_frame,resize_frame=1,show=False)
centroids = get_droplets_centroids(test_frame)
for i, c in enumerate(centroids):
    cX = c[0]
    cY = c[1]
    cv2.circle(cont_frame, (cX,cY), 1, (0,0,255), 7)
    cv2.putText(cont_frame, str(i), (cX - 4, cY - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
print(cont_frame.shape)
cont_frame = data_utils.resize_my_frame(frame=cont_frame,scale_factor=2)
cv2.imshow('centroids', cont_frame)
    


# cv2.imshow('image',frames['break0'][0])