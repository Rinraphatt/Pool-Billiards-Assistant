
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
    frame = cv.imread('./pics/pool_table_ball_4.jpg')
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

        print(whitePos)
        # print(circles[0])
        print(circles[0][whitePos][0])
        print(circles[0][whitePos][1])

        cropped_whiteZone = frame[148+circles[0][whitePos][1]-100: 148+circles[0][whitePos][1]+100, 174+circles[0][whitePos][0]-100: 174+circles[0][whitePos][0]+100]
        #cv.imshow("RealWhiteCircleZone", cropped_whiteZone)

        edges = cv.Canny(cropped_whiteZone, 130, 255)
        # Detect points that form a line
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=10, maxLineGap=100)
        # Draw the detected line segments on the original frame
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(cropped_whiteZone, (x1, y1), (x2, y2), (0, 0, 255), 1)
                # Calculate the center of two line segments

        # Find the distance between the two parallel lines
        if len(lines) == 2:
            print(lines)

            x1, y1, x2, y2 = lines[0][0]
            x3, y3, x4, y4 = lines[1][0]
            cv.circle(cropped_whiteZone, (x1, y1), 2, (0, 255, 255), -1)
            cv.circle(cropped_whiteZone, (x2, y2), 2, (0, 255, 255), -1)
            cv.circle(cropped_whiteZone, (x3, y3), 2, (0, 255, 255), -1)
            cv.circle(cropped_whiteZone, (x4, y4), 2, (0, 255, 255), -1)

            m1 = (round((x1+x3) /2) , round((y1+y3) /2)) 
            m3 = (round((x2+x4) /2) , round((y2+y4) /2))
            print(m1)
            cv.circle(cropped_whiteZone, m1, 1, (255, 255, 0), -1)
            cv.circle(cropped_whiteZone, m3, 1, (255, 0, 0), -1) 

        x1, y1 = m1
        x2, y2 = m3
        # Calculate the slope and intercept of the line
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Define the image boundaries
        height, width = cropped_whiteZone.shape[:2]
        y_top = 0
        y_bottom = height - 1
        x_left = 0
        x_right = width - 1

        # Calculate the intersection points of the line with the image boundaries
        if abs(m) > 1e-6:
            # Calculate the intersection points with the top and bottom image boundaries
            x_top = int(round((y_top - b) / m))
            x_bottom = int(round((y_bottom - b) / m))
            
            # Check if the intersection points are inside the image boundaries
            if x_top < x_left:
                x_top = x_left
                y_top = int(round(m * x_top + b))
            elif x_top > x_right:
                x_top = x_right
                y_top = int(round(m * x_top + b))
                
            if x_bottom < x_left:
                x_bottom = x_left
                y_bottom = int(round(m * x_bottom + b))
            elif x_bottom > x_right:
                x_bottom = x_right
                y_bottom = int(round(m * x_bottom + b))
        else:
            # If the line is vertical, set the intersection points to the left and right image boundaries
            x_top = x1
            x_bottom = x2
            
            if y1 < y2:
                y_top = y1
                y_bottom = y_bottom
            else:
                y_top = y2
                y_bottom = y1

            if y_top < y_top:
                y_top = y_top
                x_top = int(round((y_top - b) / m))
            elif y_top > y_bottom:
                y_top = y_bottom
                x_top = int(round((y_top - b) / m))

            if y_bottom < y_top:
                y_bottom = y_top
                x_bottom = int(round((y_bottom - b) / m))
            elif y_bottom > y_bottom:
                y_bottom = y_bottom
                x_bottom = int(round((y_bottom - b) / m))

        # Draw the line segment between the two points
        # cv.line(cropped_whiteZone, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw a continuous line from the intersection points to the image boundaries
        cv.line(cropped_whiteZone, (x_top, y_top), (x_bottom, y_bottom), (0, 255, 0), 2)



        cv.imshow("RealWhiteCircleZone", cropped_whiteZone)
        # cv.imshow("SolidCircleZone1", frame[148+circles[0][3][1]-100: 148+circles[0][3][1]+100, 174+circles[0][3][0]-100: 174+circles[0][3][0]+100])
        # cv.imshow("SolidCircleZone2", frame[148+circles[0][6][1]-100: 148+circles[0][6][1]+100, 174+circles[0][6][0]-100: 174+circles[0][6][0]+100])
        # cv.imshow("SolidCircleZone3", frame[148+circles[0][12][1]-100: 148+circles[0][12][1]+100, 17q4+circles[0][12][0]-100: 174+circles[0][12][0]+100])
        # cv.imshow("SolidCircleZone4", frame[148+circles[0][8][1]-100: 148+circles[0][8][1]+100, 174+circles[0][8][0]-100: 174+circles[0][8][0]+100])
        # cv.imshow("SolidCircleZone5", frame[148+circles[0][10][1]-100: 148+circles[0][10][1]+100, 174+circles[0][10][0]-100: 174+circles[0][10][0]+100])
        # cv.imshow("SolidCircleZone6", frame[148+circles[0][14][1]-100: 148+circles[0][14][1]+100, 174+circles[0][14][0]-100: 174+circles[0][14][0]+100])
        # cv.imshow("SolidCircleZone7", frame[148+circles[0][9][1]-100: 148+circles[0][9][1]+100, 174+circles[0][9][0]-100: 174+circles[0][9][0]+100])
        

    circleZones = []
    

    # cv.imshow("circles", frame)
    # cv.imshow("grayFrame", blurFrame)
    # cv.imshow("Output", outputDrawing)
    # cv.imshow("Detected Circles", cropped_image)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cv.destroyAllWindows()