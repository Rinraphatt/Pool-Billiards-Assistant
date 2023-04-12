
from dis import dis
from tkinter import Frame
from tkinter.colorchooser import Chooser
from turtle import circle
import cv2
import numpy as np
import math
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
#cap = cv2.VideoCapture('./videos/new1080.mp4')
# set frame rate to 30 fps
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 10)
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

# start processing loop
frame_count = 0

hsv = 0

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.imread('./pics/new5.jpg')
    # frame = cv2.resize(frame, (1920, 1080))   

    # if not ret: break
    # increment frame count
    frame_count += 1

    if frame_count % frame_interval == 0:

        # Perspective Transform
        tl = (212 ,180)
        bl = (159 ,923)
        tr = (1696 ,178)
        br = (1750 ,925)
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
        #grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # White ball detect
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of white color in HSV
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([179, 60, 255])

        # Threshold the image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Find contours of the white areas
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = (0, 0)
        # Draw the contour with the largest area, which should correspond to the pool ball
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Create a circular mask with a fixed radius of 100 pixels
            mask = np.zeros_like(frame)
            cv2.circle(mask, center, 200, (255, 255, 255), -1, cv2.LINE_AA)
            # Apply the mask to the original image using bitwise operations
            masked_img = cv2.bitwise_and(frame, mask)

            # Crop the circular region of the pool ball
            x1 = int(x - 200)
            y1 = int(y - 200)
            x2 = int(x + 200)
            y2 = int(y + 200)
            if x1 <0  :
                x1 = 1
            if y1 < 0 :
                y1 = 1
            # Cropped White Zone IMG
            print(y1,y2)
            whiteball_zone = masked_img[y1:y2, x1:x2]
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

        # Cue Detection
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(whiteball_zone, cv2.COLOR_BGR2HSV)

        # Define a cue white color threshold
        lower_white = np.array([130, 20, 100])
        upper_white = np.array([179, 255, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)
        output = cv2.bitwise_and(whiteball_zone, whiteball_zone, mask=mask)

        # Detect Edge of pool Cue
        edges = cv2.Canny(output, 200, 255)
        # Detect points that form a line
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60,
                                minLineLength=50, maxLineGap=120)
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

                m1 = (round((x1+x3) / 2), round((y1+y3) / 2))
                m3 = (round((x2+x4) / 2), round((y2+y4) / 2))
                # Convert point in Crop into Original frame
                m1 = (m1[0]+center[0]-200, m1[1]+center[1]-200)
                m3 = (m3[0]+center[0]-200, m3[1]+center[1]-200)

                dis1ToCen = math.sqrt(pow(m1[0]-center[0], 2)+pow(m1[1]-center[1], 2))
                dis2ToCen = math.sqrt(pow(m3[0]-center[0], 2)+pow(m3[1]-center[1], 2))

                # print(m1)
                cv2.circle(frame, m1, 2, (0, 0, 255), -1)
                cv2.circle(frame, m3, 2, (0, 0, 255), -1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'M1 X: '+ str(m1[0])+ " Y: "+ str(m1[1]), m1, font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'M3 X: '+ str(m3[0])+ " Y: "+ str(m3[1]), m3, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
                

                cv2.putText(frame, 'Hor : '+ hor + " Ver : "+ ver, (100, 300), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
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

                cv2.line(frame, (x_top, y_top),
                        (x_bottom, y_bottom), (0, 255, 0), 2)



                if (ver == 'up' and hor == 'left'):
                    x3,y3 = (x_top,y_top)
                    if (y_top == 0 and y_bottom == 1080):
                        cv2.line(frame, (x_top,y_top),(x_top-abs(x_bottom-x_top),1080), (0, 0, 255), 2)
                        x4,y4 = (x_top-abs(x_bottom-x_top),1080)
                        #cv2.line(frame, (x_top-abs(x_bottom-x_top),1080),(x_top-abs(x_bottom-x_top)-abs(x_top-x_top-abs(x_bottom-x_top)),0), (255, 0, 0), 2)

                    elif (x_top == 0 and x_bottom == 1920):
                        cv2.line(frame, (x_top,y_top),(1920,y_top-abs(y_top-y_bottom)), (0, 0, 255), 2)
                        x4,y4 = (1920,y_top-abs(y_top-y_bottom))
                        #cv2.line(frame, (1920,y_top-abs(y_top-y_bottom)),(0,y_top-abs(y_bottom-y_top)-abs(y_top-y_top-abs(y_bottom-y_top))), (255, 0, 0), 2)  

                if (ver == 'up' and hor == 'right'):
                    x3,y3 = (x_top,y_top)
                    if (y_top == 0 and y_bottom == 1080):
                        cv2.line(frame, (x_top,y_top),(x_top+abs(x_bottom-x_top),1080), (0, 0, 255), 2)
                        x4,y4 = (x_top+abs(x_bottom-x_top),1080)
                        #cv2.line(frame, (x_top+abs(x_bottom-x_top),1080),(x_top-abs(x_bottom-x_top)-abs(x_top-x_top-abs(x_bottom-x_top)),0), (255, 0, 0), 2)
                    elif (x_top == 0 and x_bottom == 1920):
                        cv2.line(frame, (x_top,y_top),(0,y_top-abs(y_top-y_bottom)), (0, 0, 255), 2)
                        x4,y4 = (0,y_top-abs(y_top-y_bottom))                        
                        #cv2.line(frame, (0,y_top-abs(y_top-y_bottom)),(1920,y_top-abs(y_bottom-y_top)-abs(y_top-y_top-abs(y_bottom-y_top))), (255, 0, 0), 2)     

                if (ver == 'down' and hor == 'left'):
                    x3,y3 = (x_bottom,y_bottom)
                    if (y_top == 0 and y_bottom == 1080):
                        cv2.line(frame, (x_bottom,y_bottom),(x_bottom-abs(x_bottom-x_top),0), (0, 0, 255), 2)
                        x4,y4 = (x_bottom-abs(x_bottom-x_top),0)                      
                        #cv2.line(frame, (x_bottom-abs(x_bottom-x_top),0),(x_bottom-abs(x_bottom-x_top)-abs(x_bottom-x_bottom-abs(x_bottom-x_top)),0), (255, 0, 0), 2)
                    elif (x_top == 1920 and x_bottom == 0):
                        cv2.line(frame, (x_bottom,y_bottom),(1920,y_bottom+abs(y_top-y_bottom)), (0, 0, 255), 2)   
                        x4,y4 = (1920,y_bottom+abs(y_top-y_bottom))
                if (ver == 'down' and hor == 'right'):
                    x3,y3 = (x_bottom,y_bottom)
                    if (y_top == 0 and y_bottom == 1080):
                        cv2.line(frame, (x_bottom,y_bottom),(x_bottom+abs(x_bottom-x_top),0), (0, 0, 255), 2)
                        x4,y4 = (x_bottom+abs(x_bottom-x_top),0)
                    elif (x_top == 0 and x_bottom == 1920):
                        cv2.line(frame, (x_bottom,y_bottom),(0,y_bottom+abs(y_top-y_bottom)), (0, 0, 255), 2)
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
                print(ver,hor)
                print(x3,y3)
                print(x4,y4)
                
                #real_y = mx + b
                print(m)
                print(b)
                # Create Third reflex line    
                if (ver == 'up' and hor == 'left'):
                    if (y3 == 1080 and y4 == 0):
                        if x4 < 0 :
                            cv2.line(frame, (0,int(findSlope(x3, y3, x4, y4,"y",0)[2])),(abs(x4),0), (255, 0, 0), 2)
                        else :
                            cv2.line(frame, (x4,y4),(x4-abs(x4-x3),1080), (255, 0, 0), 2)
                        #cv2.line(frame, (x_top-abs(x_bottom-x_top),1080),(x_top-abs(x_bottom-x_top)-abs(x_top-x_top-abs(x_bottom-x_top)),0), (255, 0, 0), 2)

                    elif (x3 == 1920 and x4 == 0):
                        if y4 < 0 :
                            cv2.line(frame, (int(findSlope(x3, y3, x4, y4,"x",0)[2]),0),(0,abs(y4)), (255, 0, 0), 2)
                        else :
                            cv2.line(frame, (x4,y4),(1920,y4-abs(y4-y3)), (255, 0, 0), 2)
                        #cv2.line(frame, (1920,y_top-abs(y_top-y_bottom)),(0,y_top-abs(y_bottom-y_top)-abs(y_top-y_top-abs(y_bottom-y_top))), (255, 0, 0), 2)  

                if (ver == 'up' and hor == 'right'):
                    if (y3 == 1080 and y4 == 0):
                        if x4 > 1920 :
                            cv2.line(frame, (1920,int(findSlope(x3, y3, x4, y4,"y",1920)[2])),(1920-abs(x4-1920),1080), (255, 0, 0), 2)
                        else :
                            cv2.line(frame, (x4,y4),(x4+abs(x3-x4),1080), (255, 0, 0), 2)
                        #cv2.line(frame, (x_top+abs(x_bottom-x_top),1080),(x_top-abs(x_bottom-x_top)-abs(x_top-x_top-abs(x_bottom-x_top)),0), (255, 0, 0), 2)
                    elif (x3 == 0 and x4 == 1920):
                        if y4 < 0 :
                            cv2.line(frame, (int(findSlope(x3, y3, x4, y4,"x",0)[2]),0),(1920,abs(y4)), (255, 0, 0), 2)
                        else :
                            cv2.line(frame, (x4,y4),(0,y4-abs(y4-y3)), (255, 0, 0), 2)                   
                        #cv2.line(frame, (0,y_top-abs(y_top-y_bottom)),(1920,y_top-abs(y_bottom-y_top)-abs(y_top-y_top-abs(y_bottom-y_top))), (255, 0, 0), 2)     

                if (ver == 'down' and hor == 'left'):
                    if (y3 == 0 and y4 == 1080):
                        if x4 < 0 :
                            cv2.line(frame, (0,int(findSlope(x3, y3, x4, y4,"y",0)[2])),(abs(x4),1080), (255, 0, 0), 2)  
                        else : 
                            cv2.line(frame, (x4,y4),(x4-abs(x4-x3),0), (255, 0, 0), 2)                 
                    elif (x3 == 1920 and x4 == 0):
                        if y4 > 1080 :
                            cv2.line(frame, (int(findSlope(x3, y3, x4, y4,"x",1080)[2]),1080),(0,1080-(y4-1080)), (255, 0, 0), 2)
                        else :
                            pass
                if (ver == 'down' and hor == 'right'):
                    if (y3 == 0 and y4 == 1080):
                        if x4 > 1920 :
                            cv2.line(frame, (1920,int(findSlope(x3, y3, x4, y4,"y",1920)[2])),(1920-abs(1920-x4),1080), (255, 0, 0), 2)
                        cv2.line(frame, (x4,y4),(x4+abs(x4-x3),0), (255, 0, 0), 2)

                    elif (x3 == 0 and x4 == 1920):
                        if y4 > 1080:
                            cv2.line(frame, (int(findSlope(x3, y3, x4, y4,"x",1080)[2]),1080),(1920,1080-abs(y4-1080)), (255, 0, 0), 2) 
                        else :
                            cv2.line(frame, (x4,y4),(0,y4+abs(y4-y3)), (255, 0, 0), 2)

                
                cv2.putText(frame, 'X_top: '+ str(x_top)+ " Y_top: "+ str(y_top), (100,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'X_bottom: '+ str(x_bottom)+ " Y_bottom: "+ str(y_bottom), (100,200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                print("Top : ", end=" ")         
                print(x_top, y_top)
                print("Botton : ", end=" ")    
                print(x_bottom, y_bottom)


        cv2.imshow("test", output)
        cv2.imshow('White Ball Zone', whiteball_zone)
        cv2.imshow("Cropped", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
