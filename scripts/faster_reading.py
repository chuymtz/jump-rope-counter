from threading import Thread
from queue import Queue # assuming a newish py version
import sys
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import os
import time

class FileVideoStream(object):
    def __init__(self, path, queueSize=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
  
    def start(self):
        t = Thread(target = self.update, args = ())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return None
            
            if not self.Q.full():
                ret, frame = self.stream.read()
                
                if not ret:
                    self.stop()
                    return None
                
                self.Q.put(frame)
    
    def read(self):
        return self.Q.get()
    
    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True

path2video = os.path.join('data','test2.mp4')

# stream = cv2.VideoCapture(path2video)
fvs = FileVideoStream(path2video).start()
time.sleep(1.0)
fps = FPS().start()

while fvs.more():
    
    frame = fvs.read()
    frame = imutils.resize(frame, width = 450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])
    
    m = 'Slow way'
    # m = "Queue Size: {}".format(fvs.Q.qsize())
    cv2.putText(frame, m, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    fps.update()
    
    if cv2.waitKey(1) == ord('q'):
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()

