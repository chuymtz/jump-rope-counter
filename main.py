from re import S
from shutil import ExecError
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import time
import pandas as pd

path2video = os.path.join('data','test2.mp4')
outpath = os.path.join('data','output','test.mp4')

# https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

class JumpPose(object):
    def __init__(self, frame) -> None:
        self.frame = frame
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.poses = self.mpPose.Pose(static_image_mode=True, 
                                      model_complexity=2,
                                      enable_segmentation=True,
                                      min_detection_confidence=0.5)
        
    def get_results(self):
        self.img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.results = self.poses.process(self.img)
    
    def draw_on_frame(self):
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(self)
    
class Jumper(object):
    def __init__(self, video_path:str, outpuath: str):
        self.video_path = video_path
        self.outpath = outpath
        if os.path.isfile(outpath):
            os.remove(outpath)
        self.flip_video = False

        try:
            assert os.path.isfile(path2video)
            self.cap = cv2.VideoCapture(path2video)
            self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_len_secs = self.n_frames / self.fps
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.vidWriter = cv2.VideoWriter(outpath,
                                             cv2.VideoWriter_fourcc('P','I','M','1'), 
                                             self.fps, (self.width, self.height))
            
        except Exception as e:
            print(e)
            
        self.data = []

self = Jumper(path2video, outpath)
data = []
for i in range(self.n_frames):
    print(i,' of ',self.n_frames)
    ret, frame = self.cap.read()
    if ret:
        if self.flip_video:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        p = JumpPose(frame)
        p.get_results()
        if p.results.pose_landmarks:
            p.mpDraw.draw_landmarks(frame, p.results.pose_landmarks, p.mpPose.POSE_CONNECTIONS)
        
        n, m, _ = frame.shape
        r = n/m
        w = 400
        cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('frame',frame)
        cv2.resizeWindow('frame',w,np.int32(w*r))
        
        self.vidWriter.write(frame)
        
    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()
self.cap.release()
self.vidWriter.release()
    # p = JumpPose(frame)
    # p.get_results()

    # # plt.imshow(p.img)
    # # plt.show()
    
    # mrks = p.results.pose_landmarks
    
    # if mrks is not None:
        
    #     bodypart = mrks.landmark[p.mpPose.PoseLandmark.NOSE]
    #     data.append({'bodypart': 'NOSE', 
    #                     'time': i / self.fps,
    #                     'x':bodypart.x,
    #                     'y':bodypart.y,
    #                     'z':bodypart.z,
    #                     'visibility':bodypart.visibility})
        # for j, bodypart in enumerate(mrks.landmark):
        #     data.append({'bodypart': j, 
        #                 'time': i / self.fps,
        #                 'x':bodypart.x,
        #                 'y':bodypart.y,
        #                 'z':bodypart.z,
        #                 'visibility':bodypart.visibility})
    
    
    
            
            

    
# data = pd.DataFrame(data)
    

    



























