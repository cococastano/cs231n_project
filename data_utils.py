import cv2
import numpy as np
import os
import random

def pull_frame_range(frame_range = [3], video_dir=None, num_break=None, 
                     num_nobreak=None, save_option=False):
    """
    Returns a dictionary with break or nobreak keys for the 4th frame (by 
    default) or a range of frames. There is built in randomness in the videos
    that are selected to avoid pulling the same series of videos.
    
    frame_range: list of frames of interest, should be between 0 and 56
    video_dir: directory path containing subdirectories with videos
    num_break: number of break videos ranges desired; None retrieves all break
        videos that are available
    num_nobreak: number of nobreak videos ranges desired; None retrieves all
        nobreak videos that are available
    save_option: save dictionaries to current directory if True
    """

    if video_dir == None:
        video_dir = 'C:/Users/nicas/Documents/' + \
                    'CS231N-ConvNNImageRecognition/' + \
                    'Project/droplets_raw_movies'
    

    break_files = []
    nobreak_files = []
    # generate list of paths to files in video_dir
    for subdir, dirs, files in os.walk(video_dir):
        tags = subdir.split('\\')
        for file in files:
            file_name = subdir + os.sep + file
            # lets not assume all files found are .avi video files
            if '.avi' in file:
                if tags[-1] == 'break': 
                    break_files.append(file_name)
                elif tags[-1] == 'nobreak':
                    nobreak_files.append(file_name)
                    
    # shuffle the available files
    if num_break == None: num_break = len(break_files)
    if num_nobreak == None: num_nobreak = len(nobreak_files)
    break_files = random.sample(break_files, 
                                len(break_files))[:num_break]
    nobreak_files = random.sample(nobreak_files, 
                                  len(nobreak_files))[:num_nobreak]
    
    # loop over randomized and cut lists of file names
    my_frames = {}
    break_count = 0
    nobreak_count = 0
    for i, file in enumerate(break_files + nobreak_files):
        # read video file
        vid = cv2.VideoCapture(file)
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        tags = file.split('\\')
        if tags[-2] == 'break': 
            my_key = tags[-2] + str(break_count)
            break_count += 1
            print(my_key)
        elif tags[-2] == 'nobreak':
            my_key = tags[-2] + str(nobreak_count)
            nobreak_count += 1
            print(my_key)
        for i, f in enumerate(frame_range):
            fr = 1 / num_frames * (f + 1)
            # set frame 'index' between 0.0 and 1.0 to be decoded
            
            vid.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ret, frame = vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # check indices for non black and white (gray border)
            # assume a given video has same border, not all videos
            if i == 0:
                crop_locs = np.where((np.logical_or(frame == np.amin(frame),
                                                    frame == np.amax(frame))))
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
        
        vid.release()
    
    return my_frames
        
    
# example function call                
# my_frames = pull_frame_range(frame_range=[2,3,4], num_break=9, num_nobreak=8) 
      
