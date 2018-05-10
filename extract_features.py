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
    max_contour = 300
    min_contour = 10
    orig_frame = frame
    # rgb_frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    _, my_contours, _ = cv2.findContours(orig_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centroids = []
    areas = []
    contours = []
    for contour in my_contours:
        # remove the contour of channel and the gap between drops
        if (len(contour) < max_contour) & (len(contour) > min_contour):
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

    # sort list by first element in tuple
    centroids.sort(key=lambda x: x[0], reverse=True)
    # wrap in dictionary
    centroids_out = {}
    for i, cent in enumerate(centroids):
        centroids_out[i] = cent

    return centroids_out, areas, contours

def get_n_leading_droplets_metric(frame, n, metric=None):
    """
    Return some metric for polygon made with the leading n droplets
    along with a dictionary of indexed keys for tuples with x and y
    centroid coordinates of the triangle vertices
    """
    centroids, _, _ = get_droplets_info(frame)
    constr_x_loc = data_utils.find_constriction(frame)
    leading_drops = [coord for coord in centroids.values()
                     if coord[0] < constr_x_loc]
    # keep only drops on the left of the contriction
    if n > len(leading_drops):
        print('Number of droplets being adjusted')
        n = len(leading_drops)
    # keep some number of drops to the left of the constriction
    leading_drops = leading_drops[:n]
    cc_sorted_verts = leading_drops
    leading_drops = {key: val for key, val in enumerate(leading_drops)}

    if metric == None or metric.upper() == 'AREA':
        cc_sorted_verts = data_utils.polygon_vert_sort(cc_sorted_verts)
        # implement shoelace formula to find area of polygon
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += cc_sorted_verts[i][0] * cc_sorted_verts[j][1]
            area -= cc_sorted_verts[j][0] * cc_sorted_verts[i][1]
        metric_val = abs(area) / 2.0

    return (leading_drops, metric_val)


#### test code ####
# make a video
test_frame = data_utils.pull_frame_range(frame_range=[3])
test_frame = test_frame['break0'][0]
width = test_frame.shape[0]
height = test_frame.shape[1]
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter('all_frames_mark_up.avi', fourcc, 10, (width * 2, height * 2))
n = 3  # number of vertices of polygon to interrogate
frame_range = [i for i in range(0, 4)]
frames = data_utils.pull_frame_range(frame_range=frame_range)
# do some plotting to verify what we've got so far
for frame_key in frames:
    for i, frame in enumerate(frames[frame_key]):
        # add contours to show_frame
        show_frame = data_utils.show_my_countours(frame, contour_i=-1,
                                                  resize_frame=1, show=True)
        # add centroids to show_frame
        centroids, areas, contours = get_droplets_info(test_frame)
        for c in centroids:
            cX = centroids[c][0]
            cY = centroids[c][1]
            cv2.circle(show_frame, (cX, cY), 1, (0, 0, 255), 7)
            cv2.putText(show_frame, str(c), (cX + 4, cY - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # add polygon with n vertices to show_frame
        leading_centroids, area = get_n_leading_droplets_metric(frame, n)
        leading_centroids = [(coord) for coord in leading_centroids.values()]
        leading_centroids.append(leading_centroids[0])
        leading_centroids = np.int32(np.array(leading_centroids))
        cv2.polylines(show_frame, [leading_centroids], True, (255, 60, 255))
        # add constriction location to show_frame
        constric_loc = data_utils.find_constriction(frame)
        y1 = int(frame.shape[0] / 3)
        y2 = int(frame.shape[0] / 3 * 2)
        cv2.line(show_frame, (constric_loc, y1),
                 (constric_loc, y2), (0, 150, 255), 2)
        frame_str = frame_key + ', frame ' + str(i)
        # add frame label to show_frame
        show_frame = cv2.putText(show_frame, frame_str,
                                 (show_frame.shape[1] - 250,
                                  show_frame.shape[0] - 10),
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
test_frame = test_frame['break1'][0]
leading_centroids = get_n_leading_droplets_metric(test_frame, n=3)

#    break
