import cv2
import data_utils
import numpy as np

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis
      
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow,frame, overlay=True):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx**2+fy**2)
    hsv = np.ones((h, w, 3), np.uint8) * 255
#    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
#    hsv = cv2.cvtColor(temp,cv2.COLOR_RGB2HSV)    
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*10, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if overlay is True:
        alpha = 0.2
        out = cv2.addWeighted(frame, alpha, bgr, 1 - alpha, 0)
    else:
        out = bgr
    return out


################################### SCRIPT ####################################

frame_range = [i for i in range(0,30)]
my_frames = data_utils.pull_frame_range(frame_range=frame_range,
                                        num_break=2, num_nobreak=2, 
                                        add_flip=False)

for key in my_frames:
    prev_frame = my_frames[key][0]
#    prev_frame = cv2.flip(prev_frame,0)
    constr_loc = data_utils.find_constriction(prev_frame)
    prev_frame = data_utils.crop_my_frame(prev_frame,[0,constr_loc+5,
                                                      0,prev_frame.shape[0]])
    prev_frame = data_utils.resize_my_frame(prev_frame, scale_factor=4)
    prev_frame = cv2.blur(prev_frame,(17,17))
    h,w = prev_frame.shape
    mean_flow = np.zeros((h,w,3), np.float)
    N = len(my_frames[key])

    for f, frame in enumerate(my_frames[key][1:], start=0):
        # if f != 0 and f%1 == 0: continue
#        frame = cv2.flip(frame,0)
        frame = data_utils.crop_my_frame(frame,[0,constr_loc+5,
                                                0,prev_frame.shape[0]])
        frame = data_utils.resize_my_frame(frame, scale_factor=4)
        frame = cv2.blur(frame,(17,17))
        orig_frame = frame
        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 
                                            pyr_scale=0.5, levels=3, 
                                            winsize=18, iterations=3,
                                            poly_n=3, poly_sigma=1.2, 
                                            flags=0)
        prev_frame = frame
        flow_frame = draw_hsv(flow, frame, overlay = False)
        mean_flow += flow_frame/(N-1)
        cv2.imshow('flow HSV', draw_hsv(flow,draw_flow(prev_frame,flow)))
        cv2.imshow('flow', draw_flow(prev_frame,flow))
        cv2.waitKey(500)
     
    mean_flow = np.array(np.round(mean_flow), dtype=np.uint8)
    mean_flow = data_utils.resize_my_frame(mean_flow, scale_factor=0.25)
    file_name = 'C:/Users/nicas/Documents/CS231N-ConvNNImageRecognition/' + \
                'Project/opt_flow_datasets/first_six_frames_average/' + \
                key + '_noflip.png'
    cv2.imshow('small',cv2.blur(mean_flow,(5,5)))
    cv2.waitKey(500)
#    cv2.imwrite(file_name, cv2.blur(mean_flow,(5,5)))
    
    
        

    

    


