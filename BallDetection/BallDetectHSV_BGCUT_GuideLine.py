
from dis import dis
from tkinter import Frame
from tkinter.colorchooser import Chooser
from turtle import circle
import cv2
import numpy as np
import math

cameraHeight=1080 
cameraWidth=1920

# cam = cv.VideoCapture('../Test_Perspective/newVid.mp4')
# cam = cv.VideoCapture(0, cv.CAP_DSHOW)
# cam.set(cv.CAP_PROP_FRAME_HEIGHT, cameraHeight)
# cam.set(cv.CAP_PROP_FRAME_WIDTH, cameraWidth)

# cam2 = cv.VideoCapture(0, cv.CAP_DSHOW)
# cam2.set(cv.CAP_PROP_FRAME_HEIGHT, cameraHeight)
# cam2.set(cv.CAP_PROP_FRAME_WIDTH, cameraWidth)

# success, img = cam.read()
# success2, img2 = cam2.read()

prevCircle = None
cropSize = (100, 100)

# cv.namedWindow("Python Webcam Screenshot App")

outputDrawing = np.zeros((784,1568,3), np.uint8)
mtx = np.loadtxt('./arUco/calib_data/camera_matrix.txt')
dist = np.loadtxt('./arUco/calib_data/dist_coeffs.txt')
def loadSetting():
    print("loadSetting")

    global rgbColors, grayScaleValues

    file1 = open('../Setting.txt', "r+")
    # print(file1.read())
    loadStrings = file1.readlines()
    file1.close

    print(loadStrings)

    # Load All RGB Values
    for i in range(len(rgbColors)):
        loadRGB = loadStrings[i].split()
            
        rgbColors[i][2] = float(loadRGB[2])
        rgbColors[i][1] = float(loadRGB[1])
        rgbColors[i][0] = float(loadRGB[0])

    # Load All Gray Scale Values
    for i in range(len(grayScaleValues)):
        loadGrayScaleValues = loadStrings[i+9].split()
            
        grayScaleValues[i][0] = float(loadGrayScaleValues[0])
        grayScaleValues[i][1] = float(loadGrayScaleValues[1])

    print(loadStrings)

def checkWinCondition(updatedBall):
    if not ('White' in updatedBall):
        print('Failed')
    elif ((len(updatedBall) == 1) and ('White' in updatedBall)):
        print('Success')
    else:
        print('Proceeding')

def findSlope(start_x,start_y,end_x,end_y,find=None,interest_value=None) :
    m = 0
    if (end_x-start_x) != 0:
        m = (end_y - start_y) / (end_x - start_x)
    else:
        m = (end_y - start_y) / 0.01
    b = start_y - m * start_x
    if find == "x":
        return (m,b,(interest_value-b)/m)
    elif find == "y":
        return (m,b,m*interest_value+b)
    else :
        return (m,b)

width = 1920
height = 1080
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('../videos/Level1_White1.mp4')
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

lowerColor = [
    np.array([20,150,50]), #Yellow
    np.array([110,200,125]), #Blue
    np.array([0,150,155]), #Red
    np.array([110,215,150]), #Purple
    np.array([5,150,50]), #Orange
    np.array([50,100,100]), #Green
    np.array([0,100,50]), #Crimson
    np.array([0,0,0]), #Black
    np.array([100,0,100]), #White

]
upperColor = [
    np.array([40,255,255]),
    np.array([130,255,255]),
    np.array([20,255,255]),
    np.array([145,255,255]),
    np.array([25,255,255]),
    np.array([70,255,255]),
    np.array([20,255,155]),
    np.array([179,255,40]),
    np.array([179,130,255]),
]

grayScaleValues = [
    [167.0, 170.0],
    [88.0, 121.0],
    [133.0, 153.0],
    [113.0, 147.0],
    [173.0, 187.0],
    [134.0, 150.0],
    [123.0, 139.0],
    [98.0, 0.0],
    [0.0, 225.0],
]

updatedBall = []
updatedBallPos = []
updatedBallTablePos = []
detectedBall = []
detectedBallPos = []
detectedBallTablePos = []

ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]

bgFrame = None
stageState = 0

# loadSetting()

while True:
    ret, frame = cap.read()
    ret2, frame2 = cap.read()
    frame = cv2.undistort(frame, mtx, dist)
    if not ret:
        print("Break")
        break

    frame_count += 1

    if frame_count % frame_interval == 0:
        # Perspective Transform
        tl = (252 ,21)
        bl = (174 ,906)
        tr = (1701 ,31)
        br = (1764 ,933)
        cv2.circle(frame, tl, 3, (0, 0, 255), -1)
        cv2.circle(frame, bl, 3, (0, 0, 255), -1)
        cv2.circle(frame, tr, 3, (0, 0, 255), -1)
        cv2.circle(frame, br, 3, (0, 0, 255), -1)
        cv2.line(frame, tl, bl, (0, 255, 0), 2)
        cv2.line(frame, bl, br, (0, 255, 0), 2)
        cv2.line(frame, br, tr, (0, 255, 0), 2)
        cv2.line(frame, tl, tr, (0, 255, 0), 2)
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Compute the perspective transform M
        frame = cv2.warpPerspective(frame, matrix, (width, height))
        frame = frame[200:1080,0:1920]
        showFrame = cv2.warpPerspective(frame2, matrix, (width, height))
        showFrame = showFrame[200:1080,0:1920]
        # Check if there are motion or not
        motionGrayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motionBlurFrame = cv2.GaussianBlur(motionGrayFrame, (35,35), 0)

        if bgFrame is None:
            bgFrame = motionBlurFrame

        frameDelta = cv2.absdiff(bgFrame, motionBlurFrame)
        _,thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)

        n_white_pix = np.sum(thresh == 255)

        if n_white_pix <= 100:
            print("IDLE")
            hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([50,20,40])
            upper_green = np.array([100,255,255])


            mask = cv2.inRange(hsvFrame, lower_green, upper_green)
            blurFrame = cv2.GaussianBlur(mask, (7,7), 0)

            # cropped_image = frame[148:932, 174:1742]
            # cropped_Blur_image = blurFrame[148:932, 174:1742]
            # cropped_Show_image = showFrame[148:932, 174:1742]
            cv2.imshow("CroppedShowFrame1", mask)
            circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.4, 30,
                                        param1=100, param2=20, minRadius=30, maxRadius=40)

            circleZones = []
            circleZonesColor = []
            
            if circles is not None:
                circles = np.uint16(np.around(circles))

                for i in circles[0, :]:
                    circleZoneColor = frame[int(i[1].item())-22:int(i[1].item())+22, int(i[0].item())-22:int(i[0].item())+22]
                    circleZonesColor.append(circleZoneColor)

                    cv2.circle(showFrame, (i[0], i[1]), i[2], (255,0,255), 2)

            if circles is not None:
                # Find Color and Type of Every Balls
                for i in range(len(circleZonesColor)):
                    maxSameColor = 0
                    maxSameColorPos = -1
                    semiSameColorPos = -1
                    # print('start ', maxSameColor, maxSameColorPos)

                    colorCounter = 0
                    whiteCounter = 0

                    # print('circles = ', circles)
                    # print('circleZonesColor = ', circleZonesColor)

                    # print(i, ' = ', circleZonesColor[i])
                    # print('len = ', len(circleZonesColor[i]))

                    # if circleZonesColor[i].size != 0 :
                    #     print("exist bracket = ", circleZonesColor[i])
                    # else:
                    #     print('null bracket')


                    # print('size = ', circleZonesColor[i].size)
                    if circleZonesColor[i].size != 0:
                        # print('size after = ', circleZonesColor[i].size)
                        hsvcircleZone = cv2.cvtColor(circleZonesColor[i], cv2.COLOR_BGR2HSV)
                        for j in range(len(lowerColor)):
                            mask = cv2.inRange(hsvcircleZone, lowerColor[j], upperColor[j])
                            samePixs = np.sum(mask == 255)

                            if j == 8:
                                whiteCounter = samePixs

                            if samePixs > maxSameColor:
                                semiSameColorPos = maxSameColorPos
                                maxSameColor = samePixs
                                maxSameColorPos = j
                            # print('Same to : ', j, samePixs)

                        ballType = 'none'

                        if maxSameColorPos == 8 and maxSameColor < 1800:
                            maxSameColorPos = semiSameColorPos
                            ballType = 'Stripe'
                        else:
                            if abs(maxSameColor - whiteCounter) >= 400:
                                ballType = 'Solid'
                            else:
                                ballType = 'Stripe'

                        similarColor = ''

                        if maxSameColorPos == 0:
                            similarColor = 'Yellow'
                        elif maxSameColorPos == 1:
                            similarColor = 'Blue'
                        elif maxSameColorPos == 2:
                            similarColor = 'Red'
                        elif maxSameColorPos == 3:
                            similarColor = 'Purple'
                        elif maxSameColorPos == 4:
                            similarColor = 'Orange'
                        elif maxSameColorPos == 5:
                            similarColor = 'Green'
                        elif maxSameColorPos == 6:
                            similarColor = 'Crimson'
                        elif maxSameColorPos == 7:
                            similarColor = 'Black'
                        elif maxSameColorPos == 8:
                            similarColor = 'White'

                        # print(f'Similar to {similarColor}')
                        detectedBall.append(similarColor)
                        detectedBallPos.append(maxSameColorPos)
                        detectedBallTablePos.append((circles[0][i][0], circles[0][i][1]))
                        
                        cv2.putText(showFrame, f'Number : {i}', (circles[0][i][0], circles[0][i][1]-80), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(showFrame, f'Color : {similarColor}', (circles[0][i][0], circles[0][i][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(showFrame, f'x : {circles[0][i][0]}', (circles[0][i][0], circles[0][i][1]-30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(showFrame, f'y : {circles[0][i][1]}', (circles[0][i][0], circles[0][i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        # if (maxSameColorPos >= 0 and maxSameColorPos <= 6):
                        #         cv2.putText(showFrame, f'Type : {ballType}', (circles[0][i][0], circles[0][i][1]-30), cv2.FONT_HERSHEY_SIMPLEX, 
                        #             0.7, (255, 0, 255), 2, cv2.LINE_AA)

            for k in range(len(detectedBallPos)) :
                if detectedBall[k] not in updatedBall:
                    ballProbs[detectedBallPos[k]] += 1
                    if ballProbs[detectedBallPos[k]] >= 10:
                        ballProbs[detectedBallPos[k]] = 10
                        
                    if ballProbs[detectedBallPos[k]] == 10:
                        updatedBall.append(detectedBall[k])
                        updatedBallPos.append(detectedBallPos[k])
                        updatedBallTablePos.append(detectedBallTablePos[k])

            for k in range(len(updatedBall)-1, -1, -1) :
                print('K = ', k)
                if updatedBall[k] not in detectedBall :
                    ballProbs[updatedBallPos[k]] -= 1
                    if ballProbs[updatedBallPos[k]] <= 0:
                        ballProbs[updatedBallPos[k]] = 0
                        
                    if ballProbs[updatedBallPos[k]] == 0:
                        updatedBall.pop(k)
                        updatedBallPos.pop(k)
                        updatedBallTablePos.pop(k)
                else :
                    ballProbs[updatedBallPos[k]] += 1
                    if ballProbs[updatedBallPos[k]] >= 10:
                        ballProbs[updatedBallPos[k]] = 10

            print('DetectedBall = ', detectedBall)
            print('DetectedBallPos = ', detectedBallPos)
            print('DetectedBallTablePos = ', detectedBallTablePos)
            print('UpdatedBall = ', updatedBall)
            print('UpdatedBallPos = ', updatedBallPos)
            print('UpdatedBallTablePos = ', updatedBallTablePos)
            print('BallProbs = ', ballProbs)
            whitePos = -1
            if 'White' in detectedBall:
                whitePos = detectedBall.index('White')
            # print('White pos = ', whitePos)
            whiteball_zone = np.zeros((400,400,3), np.uint8)
            
            # Have White Ball 
            if whitePos != -1:
                # print(circles[0][whitePos])

                x = circles[0][whitePos][0]
                y = circles[0][whitePos][1]
                center = (int(x), int(y))
                mask = np.zeros_like(frame)
                cv2.circle(mask, center, 200, (255, 255, 255), -1, cv2.LINE_AA)
                # Apply the mask to the original image using bitwise operations
                masked_img = cv2.bitwise_and(frame, mask)

                # Crop the circular region of the pool ball
                x1 = int(x - 200)
                y1 = int(y - 200)
                x2 = int(x + 200)
                y2 = int(y + 200)
                # Cropped White Zone IMG
                if x1 <= 0 :
                    x1 = 1
                if y1 <= 0 :
                    y1 = 1
                whiteball_zone = masked_img[y1:y2, x1:x2]
                

                # Cue Detection
                # Convert the frame to HSV color space
                print(whiteball_zone.shape[:2])
                hsv = cv2.cvtColor(whiteball_zone, cv2.COLOR_BGR2HSV)

                # Define a cue white color threshold
                lower_white = np.array([0, 60, 160])
                upper_white = np.array([100, 100, 255])

                mask = cv2.inRange(hsv, lower_white, upper_white)
                output = cv2.bitwise_and(whiteball_zone, whiteball_zone, mask=mask)

                # Detect Edge of pool Cue
                edges = cv2.Canny(output, 180, 255)
                # Detect points that form a line
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                                        minLineLength=10, maxLineGap=100)
                # Draw the detected line segments on the original frame
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        # Calculate the center of two line segments

                    # Find the distance between the two parallel lines
                    if len(lines) == 2:
                        # print(lines)

                        x1, y1, x2, y2 = lines[0][0]
                        x3, y3, x4, y4 = lines[1][0]
                        # Point of line
                        # cv.circle(cropped_whiteZone, (x1, y1), 2, (0, 255, 255), -1)
                        # cv.circle(cropped_whiteZone, (x2, y2), 2, (0, 255, 255), -1)
                        # cv.circle(cropped_whiteZone, (x3, y3), 2, (0, 255, 255), -1)
                        # cv.circle(cropped_whiteZone, (x4, y4), 2, (0, 255, 255), -1)

                        m1 = (int(round((x1+x3) / 2)), int(round((y1+y3) / 2)))
                        m3 = (int(round((x2+x4) / 2)), int(round((y2+y4) / 2)))
                        # Convert point in Crop into Original frame
                        m1 = (m1[0]+center[0]-200, m1[1]+center[1]-200)
                        m3 = (m3[0]+center[0]-200, m3[1]+center[1]-200)

                        dis1ToCen = math.sqrt(pow(m1[0]-center[0], 2)+pow(m1[1]-center[1], 2))
                        dis2ToCen = math.sqrt(pow(m3[0]-center[0], 2)+pow(m3[1]-center[1], 2))

                        # print(m1)
                        cv2.circle(showFrame, m1, 2, (0, 0, 255), -1)
                        cv2.circle(showFrame, m3, 2, (0, 0, 255), -1)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(showFrame, 'M1 X: '+ str(m1[0])+ " Y: "+ str(m1[1]), m1, font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(showFrame, 'M3 X: '+ str(m3[0])+ " Y: "+ str(m3[1]), m3, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        hor = '0'
                        ver = '0'

                        head = (0,0)
                        if dis1ToCen < dis2ToCen:
                            head = m1
                            if m1[0] < m3[0]:
                                hor = 'left'
                            else:
                                hor = 'right'

                            if m1[1] < m3[1]:
                                ver = 'up'
                            else:
                                ver = 'down'
                        else:
                            head = m3
                            if m1[0] < m3[0]:
                                hor = 'right'
                            else:
                                hor = 'left'

                            if m1[1] < m3[1]:
                                ver = 'down'
                            else:
                                ver = 'up'
                        

                        cv2.putText(showFrame, 'Hor : '+ hor + " Ver : "+ ver, (100, 300), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        x1, y1 = m1
                        x2, y2 = m3
                        x3, y3 = (0,0)
                        x4, y4 = (0,0)
                        # Calculate the slope and intercept of the line
                        m,b = findSlope(x1, y1,x2, y2)
                        y_top = 0
                        y_bottom = height
                        x_left = 0
                        x_right = width
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

                        cv2.line(showFrame, (x_top, y_top),
                                (x_bottom, y_bottom), (0, 255, 0), 2)



                        if (ver == 'up' and hor == 'left'):
                            x3,y3 = (x_top,y_top)
                            if (y_top == 0 and y_bottom == 1080):
                                cv2.line(showFrame, (x_top,y_top),(x_top-abs(x_bottom-x_top),1080), (0, 0, 255), 2)
                                x4,y4 = (x_top-abs(x_bottom-x_top),1080)
                                #cv2.line(frame, (x_top-abs(x_bottom-x_top),1080),(x_top-abs(x_bottom-x_top)-abs(x_top-x_top-abs(x_bottom-x_top)),0), (255, 0, 0), 2)

                            elif (x_top == 0 and x_bottom == 1920):
                                cv2.line(showFrame, (x_top,y_top),(1920,y_top-abs(y_top-y_bottom)), (0, 0, 255), 2)
                                x4,y4 = (1920,y_top-abs(y_top-y_bottom))
                                #cv2.line(frame, (1920,y_top-abs(y_top-y_bottom)),(0,y_top-abs(y_bottom-y_top)-abs(y_top-y_top-abs(y_bottom-y_top))), (255, 0, 0), 2)  

                        if (ver == 'up' and hor == 'right'):
                            x3,y3 = (x_top,y_top)
                            if (y_top == 0 and y_bottom == 1080):
                                cv2.line(showFrame, (x_top,y_top),(x_top+abs(x_bottom-x_top),1080), (0, 0, 255), 2)
                                x4,y4 = (x_top+abs(x_bottom-x_top),1080)
                                #cv2.line(frame, (x_top+abs(x_bottom-x_top),1080),(x_top-abs(x_bottom-x_top)-abs(x_top-x_top-abs(x_bottom-x_top)),0), (255, 0, 0), 2)
                            elif (x_top == 0 and x_bottom == 1920):
                                cv2.line(showFrame, (x_top,y_top),(0,y_top-abs(y_top-y_bottom)), (0, 0, 255), 2)
                                x4,y4 = (0,y_top-abs(y_top-y_bottom))                        
                                #cv2.line(frame, (0,y_top-abs(y_top-y_bottom)),(1920,y_top-abs(y_bottom-y_top)-abs(y_top-y_top-abs(y_bottom-y_top))), (255, 0, 0), 2)     

                        if (ver == 'down' and hor == 'left'):
                            x3,y3 = (x_bottom,y_bottom)
                            if (y_top == 0 and y_bottom == 1080):
                                cv2.line(showFrame, (x_bottom,y_bottom),(x_bottom-abs(x_bottom-x_top),0), (0, 0, 255), 2)
                                x4,y4 = (x_bottom-abs(x_bottom-x_top),0)                      
                                #cv2.line(frame, (x_bottom-abs(x_bottom-x_top),0),(x_bottom-abs(x_bottom-x_top)-abs(x_bottom-x_bottom-abs(x_bottom-x_top)),0), (255, 0, 0), 2)
                            elif (x_top == 1920 and x_bottom == 0):
                                cv2.line(showFrame, (x_bottom,y_bottom),(1920,y_bottom+abs(y_top-y_bottom)), (0, 0, 255), 2)   
                                x4,y4 = (1920,y_bottom+abs(y_top-y_bottom))
                        if (ver == 'down' and hor == 'right'):
                            x3,y3 = (x_bottom,y_bottom)
                            if (y_top == 0 and y_bottom == 1080):
                                cv2.line(showFrame, (x_bottom,y_bottom),(x_bottom+abs(x_bottom-x_top),0), (0, 0, 255), 2)
                                x4,y4 = (x_bottom+abs(x_bottom-x_top),0)
                            elif (x_top == 0 and x_bottom == 1920):
                                cv2.line(showFrame, (x_bottom,y_bottom),(0,y_bottom+abs(y_top-y_bottom)), (0, 0, 255), 2)
                                x4,y4 = (0,y_bottom+abs(y_top-y_bottom))

                        if x3 >= x4 :
                            hor = 'left'
                            if y3 >=y4:
                                ver = 'up'
                            else :
                                ver = 'down'
                        else :
                            hor = 'right'
                            if y3 >=y4:
                                ver = 'up'
                            else :
                                ver = 'down'
                        # print(ver,hor)
                        # print(x3,y3)
                        # print(x4,y4)
                        
                        #real_y = mx + b
                        # print(m)
                        # print(b)
                        # Create Third reflex line    
                        if (ver == 'up' and hor == 'left'):
                            if (y3 == 1080 and y4 == 0):
                                if x4 < 0 :
                                    cv2.line(showFrame, (0,int(findSlope(x3, y3, x4, y4,"y",0)[2])),(abs(x4),0), (255, 0, 0), 2)
                                else :
                                    cv2.line(showFrame, (x4,y4),(x4-abs(x4-x3),1080), (255, 0, 0), 2)

                            elif (x3 == 1920 and x4 == 0):
                                if y4 < 0 :
                                    cv2.line(showFrame, (int(findSlope(x3, y3, x4, y4,"x",0)[2]),0),(0,abs(y4)), (255, 0, 0), 2)
                                else :
                                    cv2.line(showFrame, (x4,y4),(1920,y4-abs(y4-y3)), (255, 0, 0), 2)

                        if (ver == 'up' and hor == 'right'):
                            if (y3 == 1080 and y4 == 0):
                                if x4 > 1920 :
                                    cv2.line(showFrame, (1920,int(findSlope(x3, y3, x4, y4,"y",1920)[2])),(1920-abs(x4-1920),1080), (255, 0, 0), 2)
                                else :
                                    cv2.line(showFrame, (x4,y4),(x4+abs(x3-x4),1080), (255, 0, 0), 2)
                            elif (x3 == 0 and x4 == 1920):
                                if y4 < 0 :
                                    cv2.line(showFrame, (int(findSlope(x3, y3, x4, y4,"x",0)[2]),0),(1920,abs(y4)), (255, 0, 0), 2)
                                else :
                                    cv2.line(showFrame, (x4,y4),(0,y4-abs(y4-y3)), (255, 0, 0), 2)                     

                        if (ver == 'down' and hor == 'left'):
                            if (y3 == 0 and y4 == 1080):
                                if x4 < 0 :
                                    cv2.line(showFrame, (0,int(findSlope(x3, y3, x4, y4,"y",0)[2])),(abs(x4),1080), (255, 0, 0), 2)  
                                else : 
                                    cv2.line(showFrame, (x4,y4),(x4-abs(x4-x3),0), (255, 0, 0), 2)                 
                            elif (x3 == 1920 and x4 == 0):
                                if y4 > 1080 :
                                    cv2.line(showFrame, (int(findSlope(x3, y3, x4, y4,"x",1080)[2]),1080),(0,1080-(y4-1080)), (255, 0, 0), 2)
                                else :
                                    cv2.line(showFrame, (x4,y4),(1920,y4-abs(y4-y3)), (255, 0, 0), 2)      
                        if (ver == 'down' and hor == 'right'):
                            if (y3 == 0 and y4 == 1080):
                                if x4 > 1920 :
                                    cv2.line(showFrame, (1920,int(findSlope(x3, y3, x4, y4,"y",1920)[2])),(1920-abs(1920-x4),1080), (255, 0, 0), 2)
                                cv2.line(showFrame, (x4,y4),(x4+abs(x4-x3),0), (255, 0, 0), 2)

                            elif (x3 == 0 and x4 == 1920):
                                if y4 > 1080:
                                    cv2.line(showFrame, (int(findSlope(x3, y3, x4, y4,"x",1080)[2]),1080),(1920,1080-abs(y4-1080)), (255, 0, 0), 2) 
                                else :
                                    cv2.line(showFrame, (x4,y4),(0,y4+abs(y4-y3)), (255, 0, 0), 2)

                        
                        cv2.putText(showFrame, 'X_top: '+ str(x_top)+ " Y_top: "+ str(y_top), (100,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(showFrame, 'X_bottom: '+ str(x_bottom)+ " Y_bottom: "+ str(y_bottom), (100,200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # print("Top : ", end=" ")         
                        # print(x_top, y_top)
                        # print("Botton : ", end=" ")    
                        # print(x_bottom, y_bottom)
            
            # checkWinCondition(updatedBall)
            detectedBall = []
            detectedBallPos = []
            detectedBallTablePos = []

            # State Checking Stage1
            if stageState == 0 :
                if 'White' in updatedBall and 'Black' in updatedBall :
                    stageState = 1
            if stageState == 1 :
                if 'Black' not in updatedBall : 
                    stageState = 2  

            # State Checking Stage2
            # if stageState == 0 :
            #     if 'White' in updatedBall and 'Black' in updatedBall :
            #         stageState = 1
            # if stageState == 1 :
            #     if 'Black' not in updatedBall : 
            #         stageState = 2  
            # if stageState == 2 :
            #     if 'White' in updatedBall and 'Black' in updatedBall :
            #         stageState = 3
            # if stageState == 3 :
            #     if 'Black' not in updatedBall : 
            #         stageState = 4  

            # State Checking Stage3
            # if stageState == 0 :
            #     if 'White' in updatedBall and 'Black' in updatedBall and 'Orange' in updatedBall:
            #         stageState = 1
            # if stageState == 1 :
            #     if 'Black' not in updatedBall : 
            #         stageState = 2  
            # if stageState == 2 :
            #     if 'White' in updatedBall and 'Black' in updatedBall and 'Orange' in updatedBall:
            #         stageState = 3
            # if stageState == 3 :
            #     if 'Black' not in updatedBall : 
            #         stageState = 4  
            

            cv2.putText(showFrame, f'State : {stageState}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Text For Stage 1
            if stageState == 0 :
                cv2.putText(showFrame, 'Please put Cue Ball and Black Ball(8)', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                            2, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(showFrame, 'at determined position.', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                            2, (0, 0, 0), 3, cv2.LINE_AA)
            elif stageState == 1 :
                cv2.putText(showFrame, 'Make cue ball hit black ball(8)', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                            2, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(showFrame, 'to sinks black ball(8) into the pocket.', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                            2, (0, 0, 0), 3, cv2.LINE_AA)
            elif stageState == 2 :
                cv2.putText(showFrame, 'Stage Success.', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                            2, (0, 0, 0), 3, cv2.LINE_AA)

            # Text For Stage 2
            # if stageState == 0 :
            #     cv2.putText(showFrame, 'Please put Cue Ball and Black Ball(8)', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'at determined position.', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            # elif stageState == 1 :
            #     cv2.putText(showFrame, 'Make cue ball hit black ball(8)', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'to sinks black ball(8) into the pocket.', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            # elif stageState == 2 :
            #     cv2.putText(showFrame, 'Nice Shot. Let try again.', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'Please put Cue Ball and Black Ball(8)', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'at determined position.', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            # elif stageState == 3 :
            #     cv2.putText(showFrame, 'Make cue ball hit black ball(8)', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'to sinks black ball(8) into the pocket.', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            # elif stageState == 4 :
            #     cv2.putText(showFrame, 'Stage Success.', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)

            # Text For Stage 3
            # if stageState == 0 :
            #     cv2.putText(showFrame, 'Please put Cue Ball, Black Ball(8)', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'and Orange Ball(5)', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'at determined position.', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            # elif stageState == 1 :
            #     cv2.putText(showFrame, 'ํYour objective is to sinks', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'black ball(8) into the pocket', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'and avoid to collide with', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'orange ball(5).', (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            # elif stageState == 2 :
            #     cv2.putText(showFrame, 'Nice Shot. Then..', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'Please put Cue Ball and Black Ball(8)', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'and Orange Ball(5)', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'at determined position.', (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            # elif stageState == 3 :
            #     cv2.putText(showFrame, 'Same objective.', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(showFrame, 'Let\'s go.', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)
            # elif stageState == 4 :
            #     cv2.putText(showFrame, 'Stage Success.', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 2, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow("CroppedShowFrame", showFrame)
            # cv2.imshow("CroppedBlurFrame", blurFrame)
            cv2.imshow('White Ball Zone', whiteball_zone)
            print('End Round')
        else:
            print("MOVING")
        # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blurFrame = cv2.GaussianBlur(grayFrame, (7, 7), 0)

        bgFrame = motionBlurFrame
        # cv2.imshow("AbsoluteDiff", frameDelta)
        # cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()