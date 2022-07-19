import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import time


path2video = os.path.join('data','test2.mp4')
outpath = os.path.join('data','output','test.mp4')

if os.path.isfile(outpath):
    os.remove(outpath)

assert os.path.isfile(path2video)

cap = cv2.VideoCapture(path2video)
flip_video = False
# mp.solutions.pose.Pose

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_len_secs = n_frames / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

vidWriter = cv2.VideoWriter(outpath,
                           cv2.VideoWriter_fourcc('P','I','M','1'),
                           fps, 
                           (width, height))

for i in range(n_frames):
    # time.sleep(1/fps)
    ret, frame = cap.read()
    
    if ret:
        if flip_video:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            
        n, m, _ = frame.shape
        r = n/m
        w = 400
        cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('frame',frame)
        cv2.resizeWindow('frame',w,np.int32(w*r))
        
        vidWriter.write(frame)
        
    if cv2.waitKey(1) == ord('q'):
        break
    
    
cv2.destroyAllWindows()
cap.release()
vidWriter.release()
    



























