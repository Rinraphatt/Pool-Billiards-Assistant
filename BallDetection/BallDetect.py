
from dis import dis
from tkinter import Frame
from tkinter.colorchooser import Chooser
from turtle import circle
import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)
prevCircle = None
cropSize = (100, 100)

# cv.namedWindow("Python Webcam Screenshot App")

outputDrawing = np.zeros((784,1568,3), np.uint8)

while True:
    # ret, frame = cam.read()
    frame = cv.imread('pool_table_ball.jpg')
    frame = cv.resize(frame, (1920, 1080))

    # if not ret: break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (7,7), 0)

    # circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 100,
    #                             param1=100, param2=30, minRadius=10, maxRadius=25)

    cropped_image = frame[148:932, 174:1742]
    cropped_Blur_image = blurFrame[148:932, 174:1742]

    circles = cv.HoughCircles(cropped_Blur_image, cv.HOUGH_GRADIENT, 1.4, 100,
                                param1=100, param2=30, minRadius=5, maxRadius=30)

    circleZones = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))

        circleCounter = 0

        # print(circles[0])

        for i in circles[0, :]:
            # cv.circle(frame, (i[0], i[1]), 1, (0,200,200), 2)
            # cv.circle(frame, (i[0], i[1]), i[2], (255,0,255), 2)

            cv.circle(frame, (i[0]+174, i[1]+148), i[2], (255,0,255), 2)
            # cv.circle(frame, (i[0], i[1]), 30, (255,0,255), 2)
            cv.circle(outputDrawing, (i[0]-300, i[1]-200), i[2], (255,0,255), 2)


            circleZone = blurFrame[148+int(i[1].item())-20:148+int(i[1].item())+20, 174+int(i[0].item())-20:174+int(i[0].item())+20]
            circleZones.append(circleZone)

            circleCounterInner = 0
            for j in circles[0, :]:
                cv.line(outputDrawing,(int(round(circles[0][circleCounterInner][0]-300)),int(round(circles[0][circleCounterInner][1]-200))),(int(round(circles[0][circleCounter][0]-300)),int(round(circles[0][circleCounter][1]-200))),(0,255,0),2)
                circleCounterInner += 1

            # cropped_image = blurFrame[i[0]-200: i[0]+200, i[1]-200: i[1]+200]
            circleCounter += 1

    cv.imshow("circles", frame)
    cv.imshow("cropped", cropped_image)
    # cv.imshow("Output", outputDrawing)

    if circles is not None:
        whiteValue = -1000000
        whiteZone = []

        whitePos = 0

        for i in range(len(circleZones)):
            testFrame = circleZones[i]
            avg_color_per_row = np.average(testFrame, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            print("CircleZones" + str(i) + " : " + str(avg_color))
            if avg_color > whiteValue : 
                whiteValue = avg_color
                whiteZone = circleZones[i]
                whitePos = i
            # cv.imshow("CircleZones" + str(i), circleZones[i])

        # print(whiteValue)

        # cv.imshow("WhiteCircleZone", whiteZone)

        # print(whitePos)
        # print(circles[0])
        # print(circles[0][whitePos][0])
        # print(circles[0][whitePos][1])

        # cropped_whiteZone = frame[148+circles[0][whitePos][1]-100: 148+circles[0][whitePos][1]+100, 174+circles[0][whitePos][0]-100: 174+circles[0][whitePos][0]+100]
        # cv.imshow("RealWhiteCircleZone", cropped_whiteZone)
        cv.imshow("SolidCircleZone1", frame[148+circles[0][3][1]-100: 148+circles[0][3][1]+100, 174+circles[0][3][0]-100: 174+circles[0][3][0]+100])
        cv.imshow("SolidCircleZone2", frame[148+circles[0][6][1]-100: 148+circles[0][6][1]+100, 174+circles[0][6][0]-100: 174+circles[0][6][0]+100])
        cv.imshow("SolidCircleZone3", frame[148+circles[0][12][1]-100: 148+circles[0][12][1]+100, 174+circles[0][12][0]-100: 174+circles[0][12][0]+100])
        cv.imshow("SolidCircleZone4", frame[148+circles[0][8][1]-100: 148+circles[0][8][1]+100, 174+circles[0][8][0]-100: 174+circles[0][8][0]+100])
        cv.imshow("SolidCircleZone5", frame[148+circles[0][10][1]-100: 148+circles[0][10][1]+100, 174+circles[0][10][0]-100: 174+circles[0][10][0]+100])
        cv.imshow("SolidCircleZone6", frame[148+circles[0][14][1]-100: 148+circles[0][14][1]+100, 174+circles[0][14][0]-100: 174+circles[0][14][0]+100])
        cv.imshow("SolidCircleZone7", frame[148+circles[0][9][1]-100: 148+circles[0][9][1]+100, 174+circles[0][9][0]-100: 174+circles[0][9][0]+100])
        

    circleZones = []
    

    # cv.imshow("circles", frame)
    # cv.imshow("grayFrame", blurFrame)
    # cv.imshow("Output", outputDrawing)
    # cv.imshow("Detected Circles", cropped_image)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cv.destroyAllWindows()