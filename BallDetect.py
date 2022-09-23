
from dis import dis
from tkinter import Frame
from tkinter.colorchooser import Chooser
from turtle import circle
import cv2 as cv
import numpy as np

# cam = cv.VideoCapture(0)
prevCircle = None
cropSize = (100, 100)

cv.namedWindow("Python Webcam Screenshot App")

while True:
    # ret, frame = cam.read()
    frame = cv.imread('pool_table_ball.jpg')

    # if not ret: break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (7,7), 0)

    # circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 100,
    #                             param1=100, param2=30, minRadius=2, maxRadius=30)

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 100,
                                param1=100, param2=30, minRadius=5, maxRadius=25)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv.circle(blurFrame, (i[0], i[1]), 1, (0,200,200), 2)
            cv.circle(blurFrame, (i[0], i[1]), i[2], (255,0,255), 2)
            # print(i[2])

            cropped_image = blurFrame[i[0]-200: i[0]+200, i[1]-200: i[1]+200]

    # cv.imshow("circles", frame)
    cv.imshow("grayFrame", blurFrame)
    cv.imshow("Detected Circles", cropped_image)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cv.destroyAllWindows()