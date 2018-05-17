import data_utils
import extract_features
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt


# plots of droplet configuration energies

# load data
#frame_range = [i for i in range(0,6)]
frame_range = [0,3,6,9]
my_frames = data_utils.pull_frame_range(frame_range=frame_range)
# construct X matrix and y vector
y = np.zeros((len(my_frames), 1))
frame_energies = {e: [] for e in range(len(frame_range))}
minimum = 100
maximum = -100
dict(zip(frame_range, []*len(frame_range)))
for i, key in enumerate(my_frames):
    print('extracting frame energies from',key)
    # my_frames[i] is a list of frames
    for f, frame in enumerate(my_frames[key]):
        energy = extract_features.configuration_energy_v2(frame)
        if energy > maximum: maximum = energy
        if energy < minimum: minimum = energy
        frame_energies[f].append(energy)
        # print('    frame %d energy: %.4g' % (f,energy))
    # classify a break as 0 and nobreak as 1
    my_class = key.split('_')[0]
    if 'nobreak' in my_class:
        y[i] = 1
    else:
        y[i] = 0

################################# PLOTTING ###################################

# plot histograms on one figure
plt.figure()
bins = np.linspace(minimum, maximum, 100)
for k, key in enumerate(frame_energies):
    label = 'frame ' + str(frame_range[k])
    plt.hist(frame_energies[key], bins, alpha=0.3, label=label)
plt.legend(loc='upper right')
plt.xlabel('Configuration energy [J]')
plt.ylabel('Count')
plt.title('Leading droplet configuration energies')
plt.show()

# plot histograms in subplots
subplot_rows = 2
subplot_cols = int(np.ceil(len(frame_range)/subplot_rows))
bins = np.linspace(minimum, maximum, 100)
subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(15,12))
subplot_fig.subplots_adjust(hspace=0.5,wspace=0.2)
temp = int(str(subplot_rows) + str(subplot_cols) +str(0))
for k, key in enumerate(frame_energies):
    temp += 1
    title = 'frame ' + str(frame_range[k])
    plt.subplot(temp)
    plt.hist(frame_energies[key], bins, alpha=0.7, label=label)
    plt.xlabel('Configuration energy [J]')
    plt.ylabel('Count')
    plt.title('Frame %d' % (frame_range[k]))

#if len(frame_range) % subplot_rows != 0:
#    for l in axs[int(k/subplot_rows-1),1].get_xaxis().get_majorticklabels():
#        l.set_visible(True)
#    print(int(k/subplot_rows)
#    subplot_fig.delaxes(axs[int(k/subplot_rows),1])

subplot_fig.show()

# plot histograms for each frame in subplots partitioned by break/nobreak
subplot_rows = 2
subplot_cols = int(np.ceil(len(frame_range)/subplot_rows))
bins = np.linspace(minimum, maximum, 70)
subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(15,12))
subplot_fig.subplots_adjust(hspace=0.5,wspace=0.2)
temp = int(str(subplot_rows) + str(subplot_cols) +str(0))
for k, key in enumerate(frame_energies):
    temp += 1
    no_break_energies = []
    break_energies = []
    for i, energy in enumerate(frame_energies[key]):
        if y[i] == 1:  # nobreak
            no_break_energies.append(energy)
        else:
            break_energies.append(energy)
    
    plt.subplot(temp)
    plt.hist(no_break_energies, bins, alpha=0.5, label='No breaks')
    plt.hist(break_energies, bins, alpha=0.5, label='Breaks')
    plt.xlabel('Configuration energy [J]')
    plt.ylabel('Count')
    plt.title('Frame %d' % (frame_range[k]))
    plt.legend()

#if len(frame_range) % subplot_rows != 0:
#    for l in axs[int(k/subplot_rows-1),1].get_xaxis().get_majorticklabels():
#        l.set_visible(True)
#    print(int(k/subplot_rows)
#    subplot_fig.delaxes(axs[int(k/subplot_rows),1])

subplot_fig.show()

##############################################################################

#n = 3  # number of vertices of polygon to interrogate
#frame_range = [i for i in range(0,4)]
#frames = data_utils.pull_frame_range(frame_range=[3],#frame_range,
#                                     num_break=20, num_nobreak=20)
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
#        print('area: ', polygon_area(leading_centroids), 'angle: ', 
#              leading_angle(leading_centroids)*180/np.pi)
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
#        cv2.waitKey(1000)
#    
#    
#    
#    centroids = extract_features.get_n_leading_droplets_centroids(frame, n=3)
#    area = extract_features.polygon_area(centroids=centroids)
#    leading_angle = extract_features.leading_angle(centroids=centroids)
#    
#    X[i,0] = LEO
#    X[i,1] = area
#    X[i,2] = leading_angle
#    
#    
#        
