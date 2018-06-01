import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math


def pull_frame_range(frame_range = [3], video_dir=None, num_break=None, 
                     num_nobreak=None, save_option=False, add_flip=True):
    """
    Returns a dictionary with break or nobreak keys for the 4th frame (by 
    default) or a list of a range of frames. There is built in randomness in 
    the videos that are selected to avoid pulling the same series of videos.
    
    frame_range: list of frames of interest, should be between 0 and 56
    video_dir: directory path containing subdirectories with videos
    num_break: number of break videos ranges desired; None retrieves all break
        videos that are available
    num_nobreak: number of nobreak videos ranges desired; None retrieves all
        nobreak videos that are available
    save_option: save dictionaries to current directory if True
    add_flip: True to double the data set by providing a refelcted frame
        too that is effectively the same
    
    example function call                
    my_frames = pull_frame_range(frame_range=[2,3,4], num_break=9, 
                                 num_nobreak=8)
    """

    if video_dir == None:
        video_dir = 'C:/Users/nicas/Documents/' + \
                    'CS231N-ConvNNImageRecognition/' + \
                    'Project/datasets'

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
    # print(len(nobreak_files) + len(break_files))
    # loop over randomized and cut lists of file names
    my_frames = {}
    break_count = 0
    nobreak_count = 0
    for i, file in enumerate(break_files + nobreak_files):
        # read video file
        vid = cv2.VideoCapture(file)
        tags = file.split('\\')
        LEO = tags[-1].split('_')[1]
        if tags[-2] == 'break': 
            my_key = tags[-2] + str(break_count) + '_LEO_' + LEO
            break_count += 1
        elif tags[-2] == 'nobreak':
            my_key = tags[-2] + str(nobreak_count) + '_LEO_' + LEO
            nobreak_count += 1
        my_frames[my_key] = []
        for i, f in enumerate(frame_range):
            # set frame index to be decoded
            vid.set(cv2.CAP_PROP_POS_FRAMES, f)
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
            frame = np.uint8(frame)
            # append frame to dictionary under respective key
            try:
                # try to append
                my_frames[my_key].append(frame)
            except:
                # except initializing a new list under a key (frame 1)
                my_frames[my_key] = [frame]
            if add_flip is True: 
                flipped_frame = cv2.flip(frame,0)
                my_frames[my_key].append(flipped_frame)
        
        vid.release()
    
    return my_frames

def get_data(num_train=2168, num_validation=400, num_test=200,
             feature_list=['LEO','area','angle'], reshape_frames=False,
             add_flip=True):
    """
    Get data. If feature list is None, raw image data will be returned
    (i.e. pixels values) as vectors of reshaped data. Set reshape_frames 
    option to True to produces matrix with each row being a frame's pixel
    values.
    
    add_flip: True to double the data set by providing a refelcted frame
        too that is effectively the same (passed to pull_frame_range method)
    
    NOTE: this is not set up to pull frames from a range for raw image 
        processing; this will extract one frame (with its flipped version if
        specified) across all videos
    """
    
    import extract_features

    # load data
    my_frames = pull_frame_range(frame_range=[3], add_flip=add_flip)
    # if add_flip is true you expect at twice as many frames
    if add_flip is True: 
        mult_fact = 2
    else:
        mult_fact = 1
    # construct X matrix and y vector
    try:
        n_features = len(feature_list)
        X_data = np.zeros((len(my_frames)*mult_fact, n_features))
        y_data = np.zeros((len(my_frames)*mult_fact, 1))
    except:
        dum_key = list(my_frames.keys())
        dum_key = dum_key[0]
        dummy = my_frames[dum_key][0]
        y_data = np.zeros((len(my_frames)*mult_fact, 1))
        
        if reshape_frames is True:
            X_data = np.zeros((len(my_frames)*mult_fact, 
                               dummy.shape[0]*dummy.shape[1]))
        elif reshape_frames is False:
            # one color channel for grayscale
            X_data = np.zeros((len(my_frames)*mult_fact, 1,
                               dummy.shape[0], dummy.shape[1]))
    
    for i, key in zip(range(0, mult_fact*len(my_frames)+1, 2), 
                      my_frames.keys()):
        for f, frame in enumerate(my_frames[key]):
            # note this will end up just taking features from the last frame
            # in the range!
            if feature_list != None:
                LEO = key.split('LEO_')[-1]
                centroids, _ = extract_features.\
                               get_n_leading_droplets_centroids(frame, n=3)
                area = extract_features.polygon_area(centroids=centroids)
                leading_angle = \
                    extract_features.leading_angle(centroids=centroids)
            
                X_data[i+f,0] = LEO
                X_data[i+f,1] = area
                X_data[i+f,2] = leading_angle
            else:
                if reshape_frames is True:
                    X_data[i+f,:] = np.reshape(frame, -1)
                elif reshape_frames is False:
                    X_data[i+f,0,:,:] = frame
                
            # classify a break as 0 and nobreak as 1
            my_class = key.split('_')[0]
            
            if 'nobreak' in my_class:
                y_data[i+f] = int(1)
            else:
                y_data[i+f] = int(0)
                    
    # make masks for partitioning data sets
    mask_train = list(range(0, num_train))
    mask_val = list(range(num_train, num_train + num_validation))
    mask_test = list(range(num_train + num_validation, 
                           num_train + num_validation + num_test))
    m = len(y_data)
    rand_i = [i for i in range(m)]
    rand_i = np.random.permutation(np.array(rand_i))
    # partition based on type of output X
    if reshape_frames is True:
        X_data = X_data[rand_i,:]
        # train set
        X_train = X_data[mask_train]
        # validation set
        X_val = X_data[mask_val]
        # test set
        X_test = X_data[mask_test]
        # reshape data to rows
        X_train = X_train.reshape(num_train, -1)
        X_val = X_val.reshape(num_validation, -1)
        X_test = X_test.reshape(num_test, -1)
    elif reshape_frames is False:
        X_data = X_data[rand_i,:,:,:]
        # train set
        X_train = X_data[mask_train,:,:,:]
        # validation set
        X_val = X_data[mask_val,:,:,:]
        # test set
        X_test = X_data[mask_test,:,:,:]
    
    # and the targets vector y
    y_data = y_data = y_data[rand_i,:]
    y_train = y_data[mask_train]
    y_val = y_data[mask_val]
    y_test = y_data[mask_test]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


    
def show_my_countours(frame, contours = -1, resize_frame=1, show=True):
    """  
    Returns a frame with the contours shown
    Inputs:
        frame: single channel 8-bit image such as the ones returned by 
               pull_frame_range and function shows contours. 
        countours: dictionary of contours to draw; default -1 is all contours 
        resize_frame: scalar to scale the frame by 
        show: True to display from function
    """
    orig_frame = frame
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        
    if contours == -1:
        _, my_contours, _ = cv2.findContours(orig_frame, cv2.RETR_LIST, 
                                             cv2.CHAIN_APPROX_NONE)
        frame_conts = cv2.drawContours(rgb_frame, my_contours, 
                                       contours, (0,255,0), 2)
    else:
        try:
            contours = list(contours.values())
        except:
            contours = [contours]
        for cont in contours:
            frame_conts = cv2.drawContours(rgb_frame, [cont], 
                                           0, (0,255,0), 2)
    if resize_frame != 1:
        out_frame = resize_my_frame(frame_conts, scale_factor=resize_frame)
    else:
        out_frame = frame_conts
    if show == True:
        cv2.imshow('contour image', out_frame)
        cv2.waitKey(500)
    
    return out_frame

def resize_my_frame(frame, scale_factor = 1):
    """
    Scale a given frame by a scale_factor
    """
    row, col, _ = frame.shape
    r = int(scale_factor*col/frame.shape[1])
    dim = (scale_factor * col, int(frame.shape[0]*r))
    out_frame = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
    
    return out_frame
 
def find_constriction(frame):
    """
    Find the x location of the constriction
    """
    step = 10
    n_rows, n_cols = frame.shape
    top_bound = np.zeros(int(round(n_cols/step+1)))
    low_bound = np.zeros(int(round(n_cols/step+1)))
    x_locs = np.zeros(int(round(n_cols/step+1)))
    for i,i_col in enumerate(range(0,n_cols,step)):
        my_slice = frame[:,i_col]
        #boundary has 0 pixel value; store to top and bottom location for an x
        zero_locs = np.where(my_slice == 0)
        top_bound[i] = np.min(zero_locs)
        low_bound[i] = np.max(zero_locs)
        x_locs[i] = i_col
    # just remove the last index to avoid runtime errors
    low_bound = np.delete(low_bound,-1)  
    top_bound = np.delete(top_bound,-1) 
    x_locs = np.delete(x_locs,-1) 
    # to top locations from the smoothed bottom for width of channel
    diff = savitzky_golay_smooth(low_bound,5,3) - \
           savitzky_golay_smooth(top_bound,5,3)
    # look for spike in the difference vector for constriction location
    deriv = np.gradient(diff)
    deriv2 = np.gradient(deriv)
    max_deriv2_i = np.argmax(deriv2)
    const_x_loc = x_locs[max_deriv2_i]
        
    return int(const_x_loc)
      
def savitzky_golay_smooth(y, window_size, order, deriv=0, rate=1):
    """
    Smoothing function
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError('window_size and order have to be of type int')
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError('window_size size must be a positive odd number')
    if window_size < order + 2:
        raise TypeError('window_size is too small for the polynomials order')
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, 
                half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve( m[::-1], y, mode='valid')

def polygon_vert_sort(corners):
    """
    Sort polygon vertices in counter clockwise order for correct shoelace 
    area formula implementation
    """
    # calculate centroid of the polygon
    n = len(corners) # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    # create a new list of corners which includes angles
    corners_with_angles = []
    for x, y in corners:
        an = (math.atan2(y - cy, x - cx) + 2.0 * math.pi) % (2.0 * math.pi)
        corners_with_angles.append((x, y, an))
    # sort it using the angles
    corners_with_angles.sort(key=lambda x: x[2])
    out = [(x, y) for x, y, a in corners_with_angles]
    # return the sorted corners w/ angles removed
#    return_val = [() for ]
#    return_val = map(lambda (x, y, an): (x, y), corners_with_angles)
    
    return out