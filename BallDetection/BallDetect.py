
from dis import dis
from tkinter import Frame
from tkinter.colorchooser import Chooser
from turtle import circle
import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)
prevCircle = None
cropSize = (100, 100)

cv.namedWindow("Python Webcam Screenshot App")

outputDrawing = np.zeros((784,1568,3), np.uint8)

while True:
    # ret, frame = cam.read()
    frame = cv.imread('pool_table_ball.jpg')
    frame = cv.resize(frame, (1920, 1080))

    # if not ret: break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (7,7), 0)

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 100,
                                param1=100, param2=30, minRadius=10, maxRadius=25)

    # circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 100,
    #                             param1=100, param2=30, minRadius=2, maxRadius=12)

    cropped_image = frame[148:932, 174:1742]

    # if circles is not None:
    #     circles = np.uint16(np.around(circles))

    #     circleCounter = 0

    #     for i in circles[0, :]:
    #         # cv.circle(frame, (i[0], i[1]), 1, (0,200,200), 2)
    #         # cv.circle(frame, (i[0], i[1]), i[2], (255,0,255), 2)

    #         cv.circle(frame, (i[0], i[1]), i[2]+10, (255,0,255), 2)
    #         # cv.circle(frame, (i[0], i[1]), 30, (255,0,255), 2)
    #         cv.circle(outputDrawing, (i[0]-300, i[1]-200), i[2], (255,0,255), 2)

    #         circleCounterInner = 0
    #         for j in circles[0, :]:
    #             cv.line(outputDrawing,(int(round(circles[0][circleCounterInner][0]-300)),int(round(circles[0][circleCounterInner][1]-200))),(int(round(circles[0][circleCounter][0]-300)),int(round(circles[0][circleCounter][1]-200))),(0,255,0),2)
    #             circleCounterInner += 1

    #         # cropped_image = blurFrame[i[0]-200: i[0]+200, i[1]-200: i[1]+200]
    #         circleCounter += 1

    cv.imshow("circles", frame)
    cv.imshow("cropped", cropped_image)
    cv.imshow("Output", outputDrawing)

    # cv.imshow("circles", frame)
    # cv.imshow("grayFrame", blurFrame)
    # cv.imshow("Output", outputDrawing)
    # cv.imshow("Detected Circles", cropped_image)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cv.destroyAllWindows()