
from dis import dis
from tkinter import Frame
from tkinter.colorchooser import Chooser
from turtle import circle
import cv2
import numpy as np
import math
import BallDetectionLib as BDLib

cameraHeight=1080 
cameraWidth=1920

prevCircle = None
cropSize = (100, 100)

width = 1920
height = 1080
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
# set frame rate to 30 fps
fps = cap.get(cv2.CAP_PROP_FPS)
print('fps = ', fps)
# frame_interval = int(fps / 5)
frame_interval = 3 #HAHAHA
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

# start processing loop
frame_count = 0

updatedBall = []
updatedBallPos = []
updatedBallTablePos = []
detectedBall = []
detectedBallPos = []
detectedBallTablePos = []

ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# loadSetting()
outputDrawing = np.zeros((1080,1920,3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Break")
        break

    frame_count += 1

    if frame_count % frame_interval == 0:

        perspectFrame =  BDLib.perspectiveTransform(frame)

        res = BDLib.getCircles(perspectFrame)

        # BDLib.createGuideline(perspectFrame, res[0], res[1], outputDrawing)
        if res[0] is not None :
            print(res)
            for i in range(len(res[0])) :
                cv2.circle(perspectFrame, (int(res[0][0][i][0]), int(res[0][0][i][1])), int(res[0][0][i][2]), (255,0,255), 2)
        
        updatedBall = []

        cv2.imshow("CroppedShowFrame", perspectFrame)
        cv2.imshow("OutputDrawing", outputDrawing)

        outputDrawing = np.zeros((1080,1920,3), np.uint8)
        print('End Round')

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()