# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True,help="path to input video file")
# args = vars(ap.parse_args())

path2video = os.path.join('data','test2.mp4')

# open a pointer to the video stream and start the FPS timer
# stream = cv2.VideoCapture(args["video"])
stream = cv2.VideoCapture(path2video)

fps = FPS().start()


# loop over frames from the video file stream
while True:
    
    ret, frame = stream.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width = 450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])
    cv2.putText(frame, 'Slow way', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    cv2.imshow('frame', frame)
    fps.update()
    
    if cv2.waitKey(1) == ord('q'):
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()

