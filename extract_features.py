import cv2
import numpy as np
import os
import random
import data_utils

def get_correct_contours(frame):
    """
    Return list of contours excluding areas too small to be a droplet and 
    non-droplet regions
    """
    size_thresh = 200
    # get contours
    _, my_temp_contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_NONE)
    # remove the two largest contours which are not drops
    my_temp_contours = sorted(my_temp_contours, key=cv2.contourArea, 
                              reverse=True)[2:]
    # remove contours below a treshold
    my_contours = [cont for cont in my_temp_contours if not 
                   cv2.contourArea(cont) < size_thresh]
    
    return my_contours

def get_droplets_centroids(frame):
    """
    Input a list of contours and return dictionary with indexed keys for 
    tuples with x and y centroid coordinates sorted by x coordinate 
    (right to left)
    """
    my_contours = get_correct_contours(frame)
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
    
def get_n_leading_droplets_centroids(frame, n):
    """
    Return a dictionary of n indexed keys to the left of the constriction
    for tuples with x and y centroid coordinates of the triangle vertices
    """
    centroids = get_droplets_centroids(frame)
    constr_x_loc = data_utils.find_constriction(frame)
    leading_drops = [coord for coord in centroids.values() 
                     if coord[0] < constr_x_loc]
    # keep only drops on the left of the contriction
    if n > len(leading_drops):
        print('Number of droplets being adjusted')
        n = len(leading_drops)
    # keep some number of drops to the left of the constriction
    leading_drops = leading_drops[:n]
    leading_drops = {key: val for key, val in enumerate(leading_drops)}     
    
    return leading_drops

#### methods to extract features ######

def polygon_area(centroids):
    """
    Accept dictionary of centroids of the type from 
    get_n_leading_droplets_centroids and implement shoelace formula to return 
    the polygon area
    """
    # unpack centroids dictionary
    cents = [val for val in centroids.values()]
    # sort vertices counter clockwise
    cc_sorted_verts = data_utils.polygon_vert_sort(cents)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += cc_sorted_verts[i][0] * cc_sorted_verts[j][1]
        area -= cc_sorted_verts[j][0] * cc_sorted_verts[i][1]
    area = abs(area) / 2.0
    
    return area

#######################################

#### test code ####
# make a video
test_frame = data_utils.pull_frame_range(frame_range=[3],
                                         num_break=1, num_nobreak=1)
test_frame = test_frame['break0'][0]
width = test_frame.shape[0]
height = test_frame.shape[1]
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#video = cv2.VideoWriter('all_frames_mark_up.avi', fourcc,10,(width*2,height*2))
video = cv2.VideoWriter('all_frames_mark_up.avi',-1,20,(width*2,height*2))

n = 3  # number of vertices of polygon to interrogate
frame_range = [i for i in range(0,4)]
frames = data_utils.pull_frame_range(frame_range=[3],#frame_range,
                                     num_break=1, num_nobreak=1)
# do some plotting to verify what we've got so far
for frame_key in frames:
    for i, frame in enumerate(frames[frame_key]):      
        # add contours to show_frame
        show_frame = data_utils.show_my_countours(frame,contour_i=-1,
                                                  resize_frame=1,show=False)
        # add centroids to show_frame
        centroids = get_droplets_centroids(frame)
        for c in centroids:
            cX = centroids[c][0]
            cY = centroids[c][1]
            cv2.circle(show_frame, (cX,cY), 1, (0,0,255), 7)
            cv2.putText(show_frame, str(c), (cX + 4, cY - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # add polygon with n vertices to show_frame
        leading_centroids = get_n_leading_droplets_centroids(frame,n)
        print('area: ', polygon_area(leading_centroids))
        leading_centroids = [(coord) for coord in leading_centroids.values()]
        leading_centroids.append(leading_centroids[0])
        leading_centroids = np.int32(np.array(leading_centroids))        
        cv2.polylines(show_frame, [leading_centroids], True, (255,60,255))
        # add constriction location to show_frame
        constric_loc = data_utils.find_constriction(frame)
        y1 = int(frame.shape[0]/3)
        y2 = int(frame.shape[0]/3*2)
        cv2.line(show_frame, (constric_loc, y1), 
                 (constric_loc, y2), (0,150,255), 2)
        frame_str = frame_key + ', frame ' + str(i)
        # add frame label to show_frame
        show_frame = cv2.putText(show_frame, frame_str, 
                                 (show_frame.shape[1]-250,
                                 show_frame.shape[0]-10),
                                 cv2.FONT_HERSHEY_COMPLEX, 
                                 0.6, (0, 0, 0), 2)
        # resize show_frame
        show_frame = data_utils.resize_my_frame(frame=show_frame,
                                                scale_factor=2)
        # show show_frame
        video.write(show_frame)
        cv2.imshow('mark up', show_frame)
        cv2.waitKey(50)

video.release()
       
test_frame = data_utils.pull_frame_range(frame_range=[3])
test_frame = test_frame['break0'][0]
leading_centroids = get_n_leading_droplets_centroids(test_frame, n=3)

        
#    break
    