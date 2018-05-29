import data_utils
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


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
    Return the energy associated with a configuration of n drops left of the 
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
    Return the energy associated with a configuration of n drops left of the 
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

def mean_deformation(frame, n=3):
    """ 
    Return the mean energy associated with a configuration of n drops left 
    of the constriction
    """
    # collect n centroids and contours left of constriction
    _ , contours = get_n_leading_droplets_centroids(frame, n)
    contour_list = list(contours.values())
    
    mean_deformation = 0
    for cont in contour_list:
        perim = cv2.arcLength(cont,True)
        area = cv2.contourArea(cont)
        mean_deformation += perim / (2 * np.sqrt(np.pi*area))
        
    mean_deformation = mean_deformation/n
    return mean_deformation

def sum_perimeters(frame, n=3):
    """ 
    Return the mean energy associated with a configuration of n drops left 
    of the constriction
    """
    # collect n centroids and contours left of constriction
    _ , contours = get_n_leading_droplets_centroids(frame, n)
    contour_list = list(contours.values())
    
    perim_sum = 0
    for cont in contour_list:
        perim_sum += cv2.arcLength(cont,True)
        
    return perim_sum

def outer_perimeter(frame, return_frame=False, n=3):
    """ 
    Return the sum of the perimeters of the n drops left of the constriction
    """
    # collect n centroids and contours left of constriction
    _ , contours = get_n_leading_droplets_centroids(frame, n)
    contour_list = list(contours.values())
    # make dummy frame for superimpose contour onto
    blk_im = np.ones((frame.shape[0],frame.shape[1],3),np.uint8)
    # draw contours of interest on dummy frame
    for cont in contour_list:
        blk_im = cv2.drawContours(blk_im,[cont],0,(0,255,0),-1)
    # make kernels for dilating/eroding and closing
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    # dilate image
    dil_im = cv2.dilate(blk_im, kernel_dil, iterations=1)
    # close image
    close_im = cv2.morphologyEx(dil_im, cv2.MORPH_CLOSE, kernel_close)
    # erode initial dilating
    erod_im = cv2.erode(close_im, kernel_dil, iterations=1)
    
    # get contour of processed droplets
    temp_im = cv2.cvtColor(erod_im, cv2.COLOR_BGR2GRAY)
    _, temp_im = cv2.threshold(temp_im,127,255,0)
    _, my_contours, _ = cv2.findContours(temp_im, cv2.RETR_LIST, 
                                         cv2.CHAIN_APPROX_NONE)
    # if multiple contours add perimeters
    total_perim = 0
    if len(my_contours) > 1:
        for cont in my_contours:
            total_perim += cv2.arcLength(cont,True)
    else:
        total_perim = cv2.arcLength(my_contours[0],True)

    
    # check if return markedup frame wanted
    if return_frame == False:
        out_im = None
    else:
        blk_out_im = np.ones((frame.shape[0],frame.shape[1],3),np.uint8)
        final_cont_im = cv2.drawContours(blk_out_im,my_contours,-1,
                                         (0,0,255),-1)
        alpha = 0.7
        cont_im = data_utils.show_my_countours(frame,contours=contours,
                                               show=False)
        out_im = cv2.addWeighted(final_cont_im, alpha, cont_im, 1 - alpha, 0)
    
    return (total_perim, out_im)


        
    