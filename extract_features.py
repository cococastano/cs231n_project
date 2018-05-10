import cv2
import numpy as np
import os
import random
import data_utils

def get_droplets_centroids(frame):
    """
    Input a frame and return dictionary with indexed keys for tuples with x 
    and y centroid coordinates sorted by x coordinate (left to right) and 
    with non-droplet regions removed
    """
    orig_frame = frame
    # get contours
    _, my_contours, _ = cv2.findContours(orig_frame, cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_NONE)
    # remove the two largest contours which are not drops
    my_contours = sorted(my_contours, key=cv2.contourArea, 
                         reverse=True)[2:]
    # find centroids from the moments
    centroid_list = [0] * len(my_contours)
    for i, cont in enumerate(my_contours):
        M = cv2.moments(cont)
        if M['m00'] != 0: 
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX = 0
            cY = 0
        centroid_list[i] = (cX, cY)
    # sort list by first element in tuple
    centroid_list.sort(key=lambda x: x[0], reverse=True)
    # wrap in dictionary
    centroids = {}
    for i, cent in enumerate(centroid_list):
        centroids[i] = cent        
        
    return centroids
    
def get_three_droplet_leading_angle():
    return angle


#### test code ####
frame_range = [i for i in range(0,4)]
frames = data_utils.pull_frame_range(frame_range=frame_range)
# do some plotting to verify what we've got so far
for frame_key in frames:
    for i, frame in enumerate(frames[frame_key]):        
        show_frame = data_utils.show_my_countours(frame,contour_i=-1,
                                                  resize_frame=1,show=False)
        centroids = get_droplets_centroids(frame)
        for c in centroids:
            cX = centroids[c][0]
            cY = centroids[c][1]
            cv2.circle(show_frame, (cX,cY), 1, (0,0,255), 7)
            cv2.putText(show_frame, str(c), (cX + 4, cY - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        constric_loc = data_utils.find_constriction(frame)
        y1 = int(frame.shape[0]/3)
        y2 = int(frame.shape[0]/3*2)
        cv2.line(show_frame, (constric_loc, y1), 
                 (constric_loc, y2), (0,150,255), 2)
        frame_str = frame_key + ' frame ' + str(i)
        show_frame = cv2.putText(show_frame, frame_str, 
                                 (show_frame.shape[1]-250,
                                 show_frame.shape[0]-10),
                                 cv2.FONT_HERSHEY_COMPLEX, 
                                 0.6, (0, 0, 0), 2)
        show_frame = data_utils.resize_my_frame(frame=show_frame,
                                                scale_factor=2)
        cv2.imshow('mark up', show_frame)
        cv2.waitKey(200)
    
    
#frames = data_utils.pull_frame_range(frame_range=[3])
#test_frame = frames['break0'][0]
#dummy = data_utils.find_constriction(test_frame)
#print(dummy)
#
#cont_frame = data_utils.show_my_countours(test_frame,contour_i=-1,
#                                          resize_frame=1,show=False)
#centroids = get_droplets_centroids(test_frame)
#for c in centroids:
#    cX = centroids[c][0]
#    cY = centroids[c][1]
#    cv2.circle(cont_frame, (cX,cY), 1, (0,0,255), 7)
#    cv2.putText(cont_frame, str(c), (cX - 4, cY - 4),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#cont_frame = data_utils.resize_my_frame(frame=cont_frame,scale_factor=2)
#cv2.imshow('centroids', cont_frame)
#    


# cv2.imshow('image',frames['break0'][0])