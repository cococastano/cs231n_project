# Standard imports
import cv2
#import tensorflow as tf
import numpy as np
import os

"""
This script writes all frames to a dictionary and saves the dictionary in the 
current directory. NOTE the final dictioanry file is very large if you include
many videos to save, it WILL slow down your computer. Take this as a first 
version of a method to pull the some range of frames (coming soon).

Saved dictionary:
    my_frames = {'break0': [[frame0],[frame1], ...],
                 'break1': [[frame0],[frame1],...],
                 'nobreak0': [[frame0],[frame1],...],...}
    
master_dir: directory with video subdirectories and files, for example:
    master_dir
        set01
            600ulhr_1
                break
                    LEO_*.avi
                    LEO_*.avi
                nobreak
                    ...
            600ulhr_1_2
            600ulhr_2
        set02
        ...
        
If you want to open and pull from some 'set', move it into master_dir
"""

master_dir = 'Video1'


# dictionary to hold frames with keys break1, nobreak1, break2, etc.
my_frames = {}
total_vid_count = 0
break_count = 0
nobreak_count = 0

for subdir, dirs, files in os.walk(master_dir):
    vid_count = 0
    tags = subdir.split('/')
    if vid_count != 0: print('%d videos read in subdir %s' 
                             % (vid_count, tags[-2:]))
    # loop over all files below master directory (only touches files not dirs)
    for file in files:
        vid_count += 1
        total_vid_count += 1
        
        file_name = subdir + os.sep + file
        # read a file if possible (returns None if path not a video)
        vid = cv2.VideoCapture(file_name)
        ret = True        
        
        # make key for dictionary and update (no)break counts
        if tags[-1] == 'break': 
            my_key = tags[-1] + str(break_count)
            break_count += 1
        elif tags[-1] == 'nobreak':
            my_key = tags[-1] + str(nobreak_count)
            nobreak_count += 1
        
        while ret:
            
            frame_count = 0
            ret, frame = vid.read()
            # if a valid video file
            if ret: 
                frame_count += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # check indices for non black and white (gray border)
                if frame_count == 1:
                    crop_locs = np.where((np.logical_or(frame == np.amin(frame), frame == np.amax(frame))))
                    r1 = crop_locs[0][0]
                    r2 = crop_locs[0][-1] + 1
                    c1 = crop_locs[1][0]
                    c2 = crop_locs[1][-1] + 1
                
                # crop out the gray border and normalize the array
                frame = frame[r1:r2, c1:c2]
                # expect the pixel values to be 0 or 255 (some max) so...
                frame = frame / np.amax(frame)
                
                # append frame to dictionary under respective key
                try:
                    # try to append
                    my_frames[my_key].append(frame)
                except:
                    # except initializing a new list under a key (frame 1)
                    my_frames[my_key] = [frame]               
                
#                # uncomment block to show frame (window appears in background)
#                cv2.imshow('frame', frame)
#                k = cv2.waitKey(500)  # 500 ms wait            
               
        vid.release()        
#        print('%d frames in %s' % (frame_count, tags[-2:]))
    
    # end of looping over directories in droplets_raw_movies
print('total number of videos read: %d' % (total_vid_count))
print('%d break videos, %d no break videos' % (break_count, nobreak_count))

# saving dictionary (this is a very large file)
print('saving dictionary')
np.save('my_frames.npy', my_frames)
# dummy = np.load('my_frames.npy')
