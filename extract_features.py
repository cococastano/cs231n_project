import data_utils
import cv2
import numpy as np
import os
import random


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
    Input a list of contours and return two dictionaries:
        centroids: indexed keys for tuples with x and y centroid coordinates 
        sorted by x coordinate (right to left); 1 is just left of constriction
        contours: the contours corresponding to the centroids
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
    centroid_list, contour_list = (list(x) for x in zip(
            *sorted(zip(centroid_list, my_contours), 
                    key=lambda x: x[0][0], reverse=True)))
    # centroid_list.sort(key=lambda x: x[0], reverse=True)
    # wrap in dictionary
    centroids = {}
    contours = {}
    for i, cent in enumerate(centroid_list):
        centroids[i] = cent    
        contours[i] = contour_list[i]
        
    return (centroids,contours)
    
def get_n_leading_droplets_centroids(frame, n):
    """
    Input frame and number of leading droplets of interest and return:
        centroids: a dictionary of n indexed keys to the left of the 
        constriction for tuples with x and y centroid coordinates of the 
        triangle vertices
        contours: the contours corresponding to the centroids
    """
    centroids, contours = get_droplets_centroids(frame)
    # find constriction and droplets to the left
    constr_x_loc = data_utils.find_constriction(frame)
    leading_drops_centroids = []
    leading_drops_contours = []
    leading_drops_centroids = [coord for coord in centroids.values() 
                               if coord[0] < constr_x_loc]
    leading_drops_contours = [contours[keys[1]] for keys in 
                              zip(centroids,contours) 
                              if centroids[keys[0]][0] < constr_x_loc]
    # keep only drops on the left of the contriction
    if n > len(leading_drops_centroids):
        print('Number of droplets being adjusted')
        n = len(leading_drops_centroids)
    # keep some number of drops to the left of the constriction
    leading_drops_centroids = leading_drops_centroids[:n]
    leading_drops_centroids = {key: val for key, val 
                               in enumerate(leading_drops_centroids)}
    leading_drops_contours = leading_drops_contours[:n]
    leading_drops_contours = {key: val for key, val 
                              in enumerate(leading_drops_contours)}     
    return (leading_drops_centroids, leading_drops_contours)

#### methods to extract features ######

def polygon_area(centroids):
    """
    Accept dictionary of centroids of the type from 
    get_n_leading_droplets_centroids and implement shoelace formula to return 
    the polygon area
    """
    # unpack centroids dictionary
    cents = [val for val in centroids.values()]
    n = len(cents)
    # sort vertices counter clockwise
    cc_sorted_verts = data_utils.polygon_vert_sort(cents)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += cc_sorted_verts[i][0] * cc_sorted_verts[j][1]
        area -= cc_sorted_verts[j][0] * cc_sorted_verts[i][1]
    area = abs(area) / 2.0
    
    return area

def leading_angle(centroids):
    """
    Return the angle at first vertex of polygon in radians
    """
    # unpack centroids dictionary (order preserved)
    cents = [np.array(val) for val in centroids.values()]
    a = cents[1] - cents[0]
    b = cents[2] - cents[0]
    leading_angle = np.arccos((a.dot(b))/
                              (np.linalg.norm(a)*np.linalg.norm(b)))
    return leading_angle

def configuration_energy(frame, n=3):
    """ 
    Return the energy associated witha configuration of n drops left of the 
    constriction
    """
    # hardcode conversion from pixels to microns
    cf = 30e-6/33.5  # in meters
    # surface tension sigma
    sigma = 26.65/1000.0  # N/m
    centroids, contours = get_n_leading_droplets_centroids(frame, n)
    # centroid_area = polygon_area(centroids)
    # add the first contour to the end
    cont_keys = list(contours.keys())
    contours['dummy'] = contours[cont_keys[-1]]
    cent_keys = list(centroids.keys())
    centroids['dummy'] = centroids[cent_keys[-1]]
    energy = 0
    contour_list = list(contours.values())
    centroid_list = list(centroids.values())
    for i in range(len(contour_list)-1):
        A1 = cv2.contourArea(contour_list[i])
        A2 = cv2.contourArea(contour_list[i+1])
        R1_mean = np.sqrt(A1/np.pi) * cf
        R2_mean = np.sqrt(A2/np.pi) * cf
        dist_vector = np.array(centroid_list[i+1]) - np.array(centroid_list[i])
        centroid_dist = np.linalg.norm(dist_vector) * cf
        energy += 0.5* sigma/2* ((R1_mean + R2_mean) - centroid_dist)**2
   
    return energy

def configuration_energy_v2(frame, n=3):
    """ 
    Return the energy associated witha configuration of n drops left of the 
    constriction
    """
    # hardcode conversion from pixels to microns
    cf = 30e-6/33.5  # in meters
    # surface tension sigma
    sigma = 26.65/1000.0  # N/m
    centroids, contours = get_n_leading_droplets_centroids(frame, n)
    area = polygon_area(centroids)
    angle = leading_angle(centroids)*180/np.pi
    # add the first contour to the end if isnt an in-line configuration
    if (angle < 100 and angle > 20) and (area < 900 and area > 600):
        # case where centroids are not in-line
        cont_keys = list(contours.keys())
        contours['dummy'] = contours[cont_keys[-1]]
        cent_keys = list(centroids.keys())
        centroids['dummy'] = centroids[cent_keys[-1]]
        energy = 0
        contour_list = list(contours.values())
        centroid_list = list(centroids.values())
    else:
        contour_list = list(contours.values())
        centroid_list = list(centroids.values())
        # reorder centroid labels from top to bottom
        centroid_list, contour_list = (list(x) for x in zip(
            *sorted(zip(centroid_list, contour_list), 
                    key=lambda x: x[0][1], reverse=False)))
    
    energy = 0
    for i in range(len(contour_list)-1):
        A1 = cv2.contourArea(contour_list[i])
        A2 = cv2.contourArea(contour_list[i+1])
        R1_mean = np.sqrt(A1/np.pi) * cf
        R2_mean = np.sqrt(A2/np.pi) * cf
        dist_vector = np.array(centroid_list[i+1]) - np.array(centroid_list[i])
        centroid_dist = np.linalg.norm(dist_vector) * cf
        energy += 0.5* sigma/2* ((R1_mean + R2_mean) - centroid_dist)**2
        
    return energy

#######################################

##### test code ####
## make a video
#
#test_frame = data_utils.pull_frame_range(frame_range=[3],
#                                         num_break=1, num_nobreak=1)
#test_frame = test_frame[list(test_frame.keys())[0]][0]
#width = test_frame.shape[0]
#height = test_frame.shape[1]
##fourcc = cv2.VideoWriter_fourcc(*'DIVX')
##video = cv2.VideoWriter('all_frames_mark_up.avi', fourcc,10,(width*2,height*2))
##video = cv2.VideoWriter('all_frames_mark_up.avi',-1,20,(width*2,height*2))
#
#n = 3  # number of vertices of polygon to interrogate
#frame_range = [i for i in range(0,21)]
#frames = data_utils.pull_frame_range(frame_range=frame_range,#frame_range,
#                                     num_break=5, num_nobreak=5)
## do some plotting to verify what we've got so far
#for frame_key in frames:
#    for i, frame in enumerate(frames[frame_key]):      
#        # add contours to show_frame
#        show_frame = data_utils.show_my_countours(frame,contour_i=-1,
#                                                  resize_frame=1,show=False)
#        # add centroids to show_frame
#        centroids, _ = get_droplets_centroids(frame)
#        for c in centroids:
#            cX = centroids[c][0]
#            cY = centroids[c][1]
#            cv2.circle(show_frame, (cX,cY), 1, (0,0,255), 7)
#            cv2.putText(show_frame, str(c), (cX + 4, cY - 4),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#        # add polygon with n vertices to show_frame
#        leading_centroids, _ = get_n_leading_droplets_centroids(frame,n)
#        print('area: ', polygon_area(leading_centroids), '\t angle: ', 
#              leading_angle(leading_centroids)*180/np.pi,
#              '\t frame key: ', frame_key)
#        leading_centroids = [(coord) for coord in leading_centroids.values()]
#        leading_centroids.append(leading_centroids[0])
#        leading_centroids = np.int32(np.array(leading_centroids))        
#        cv2.polylines(show_frame, [leading_centroids], True, (255,60,255))
#        # add constriction location to show_frame
#        constric_loc = data_utils.find_constriction(frame)
#        y1 = int(frame.shape[0]/3)
#        y2 = int(frame.shape[0]/3*2)
#        cv2.line(show_frame, (constric_loc, y1), 
#                 (constric_loc, y2), (0,150,255), 2)
#        frame_str = frame_key.split('_')[0]
#        frame_str = frame_key + ', frame ' + str(i)
#        # add frame label to show_frame
#        show_frame = cv2.putText(show_frame, frame_str, 
#                                 (show_frame.shape[1]-250,
#                                 show_frame.shape[0]-10),
#                                 cv2.FONT_HERSHEY_COMPLEX, 
#                                 0.5, (0, 0, 0), 2)
#        # resize show_frame
#        show_frame = data_utils.resize_my_frame(frame=show_frame,
#                                                scale_factor=2)
#        # show show_frame
#        # video.write(show_frame)
#        cv2.imshow('mark up', show_frame)
#        cv2.waitKey(700)

#video.release()
##       
test_frame = data_utils.pull_frame_range(frame_range=[3])
test_frame = test_frame[list(test_frame.keys())[0]][0]

leading_centroids, _ = get_n_leading_droplets_centroids(test_frame, n=3)
dummy = leading_angle(leading_centroids)
energy = configuration_energy(test_frame)

        
    