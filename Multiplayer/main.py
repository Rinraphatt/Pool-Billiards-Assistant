import cv2
import numpy as np
import time
import math


def findSlope(start_x, start_y, end_x, end_y, find=None, interest_value=None):
    m = 0
    if (end_x-start_x) != 0:
        m = (end_y - start_y) / (end_x - start_x)
    else:
        m = (end_y - start_y) / 0.01
    b = start_y - m * start_x
    if find == "x":
        return (m, b, (interest_value-b)/m)
    elif find == "y":
        return (m, b, m*interest_value+b)
    else:
        return (m, b)

def showLine(frame, start, end , width = 2):
    if isShowline is True :
        cv2.line(frame, start, end, (0, 0, 255), width)

def realPosition(x,y):
    circle_pos_on_img = (int(roi_x + x), int(roi_y + y))

    homogeneous_coord = np.array([circle_pos_on_img[0], circle_pos_on_img[1], 1]).reshape(-1, 1)
    original_coord = np.matmul(M_inv, homogeneous_coord)
    original_coord /= original_coord[2]
    # The resulting original coordinate is (x_o, y_o)
    x_o = int(original_coord[0][0])
    y_o = int(original_coord[1][0])

    return x_o,y_o

# lowerColor = [
#     np.array([0,0,200]),     #White
#     np.array([100,0,0]),     #Black
#     np.array([25,60,130]),   #Yellow
#     np.array([100,30,90]), #Blue
#     np.array([0,25,220]),   #Red
#     np.array([125,90,130]), #Purple
#     np.array([15,110,100]),    #Orange
#     np.array([75,100,90]),  #Green
#     np.array([0,100,50]),    #Crimson



# ]
# upperColor = [
#     np.array([179,100,255]), #White
#     np.array([179,100,245]),  #Black
#     np.array([40,255,255]),  #Yellow
#     np.array([125,255,255]), #Blue
#     np.array([10,255,255]),  #Red
#     np.array([135,255,255]), #Purple
#     np.array([25,245,255]),  #Orange
#     np.array([95,255,255]),  #Green
#     np.array([20,255,155]),  #Crimson


# ]


lowerColor = [
    np.array([20,150,50]), #Yellow
    np.array([110,200,125]), #Blue
    np.array([0,150,155]), #Red
    np.array([110,215,150]), #Purple
    np.array([5,150,50]), #Orange
    np.array([95,100,100]), #Green
    np.array([0,100,50]), #Crimson
    np.array([0,0,0]), #Black
    np.array([100,0,130]), #White

]
upperColor = [
    np.array([40,255,255]),
    np.array([130,255,255]),
    np.array([20,255,255]),
    np.array([145,255,255]),
    np.array([25,255,255]),
    np.array([110,255,255]),
    np.array([20,255,155]),
    np.array([179,255,40]),
    np.array([179,180,255]),
]


# Load the camera matrix and distortion coefficients from the calibration file
mtx = np.loadtxt('./arUco/calib_data/camera_matrix.txt')
dist = np.loadtxt('./arUco/calib_data/dist_coeffs.txt')
cap = cv2.VideoCapture(0)
width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fps = cap.get(cv2.CAP_PROP_FPS)
print('fps = ', fps)
tl = (245 ,10)
bl = (180 ,900)
tr = (1717 ,22)
br = (1760 ,930)
table_width = 1920
table_height = 880
roi_x, roi_y = 0, 200
roi_w, roi_h = 1920, 880
isShowline = True
avg_center_x = []
count_shot = 0
ball_move = False
cue_move = False
updatedBall = []
updatedBallPos = []
updatedBallTablePos = []
detectedBall = []
detectedBallPos = []
detectedBallTablePos = []

ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
while True:
    succuess, frame = cap.read()
    black_bg = np.zeros((table_height, table_width, 3), np.uint8)
    whiteball_zone = np.zeros((400, 400, 3), np.uint8)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    # Load the image to be projected
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    original_frame = frame.copy()
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    M_inv = cv2.invert(M)[1]
    # Compute the perspective transform M
    transformed_frame = cv2.warpPerspective(frame, M, (width, height))
    projection_frame = transformed_frame.copy()
    table_frame = projection_frame[200:1080,0:1920]
    
    hsv = cv2.cvtColor(table_frame, cv2.COLOR_BGR2HSV)
    blurFrame = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    # Blue
    # lower_white = np.array([0, 200, 140])
    # upper_white = np.array([179, 255, 255])
    # White with light room
    lower_white = np.array([0,0,200])
    upper_white = np.array([179,100,255])
    mask = cv2.inRange(blurFrame, lower_white, upper_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    inv_mask = cv2.bitwise_not(mask)
    output = cv2.bitwise_and(table_frame,table_frame, mask= inv_mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # Detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=20, minRadius=23, maxRadius=35)

    circleZones = []
    circleZonesColor = []
    if circles is not None :
        circles = np.round(circles[0, :]).astype("int")
        i=0
        if len(circles) <= 20 :
            for (x, y, r) in circles:
                i+=1
                # Find Position on Original Frame
                x_o,y_o = realPosition(x, y)
                x1 = x_o - 30
                y1 = y_o - 30
                x2 = x_o + 30
                y2 = y_o + 30
                if x1 < 0:
                    x1 = 1
                if y1 < 0:
                    y1 = 1
                circleZoneColor = original_frame[y1:y2, x1:x2]
                circleZonesColor.append(circleZoneColor)

                #cv2.imshow("test"+str(i),circleZoneColor)
                cv2.circle(original_frame, (x_o,y_o), r, (0,255,0), 2)


    if circles is not None:
        for i in range(len(circleZonesColor)):
            maxSameColor = 0
            maxSameColorPos = -1
            semiSameColorPos = -1
            colorCounter = 0
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

                detectedBall.append(similarColor)
                detectedBallPos.append(maxSameColorPos)
                detectedBallTablePos.append((circles[i][0], circles[i][1]))
                
                cv2.putText(table_frame, f'Number : {i}', (circles[i][0], circles[i][1]-80), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(table_frame, f'Color : {similarColor}', (circles[i][0], circles[i][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(table_frame, f'X : {circles[i][0]}', (circles[i][0], circles[i][1]-30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(table_frame, f'Y : {circles[i][1]}', (circles[i][0], circles[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 0, 255), 2, cv2.LINE_AA)

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
                updatedBallTablePos[k] = detectedBallTablePos[detectedBall.index(updatedBall[k])] 

    #print('DetectedBall = ', detectedBall)
    whitePos = -1
    blackPos = -1
    if 'White' in detectedBall:
        whitePos = detectedBall.index('White')
        detectedBall.pop(detectedBall.index('White'))
    if 'Black' in detectedBall:
        blackPos = detectedBall.index('Black')
        detectedBall.pop(detectedBall.index('Black'))

    if whitePos != -1:

        x = circles[whitePos][0]
        y = circles[whitePos][1]
        center = (int(x), int(y))
        cv2.circle(black_bg, center, 200, (255, 255, 255), 5, cv2.LINE_AA)
        x_o,y_o = realPosition(x, y)
        x1 = x_o - 200
        y1 = y_o - 200
        x2 = x_o + 200
        y2 = y_o + 200
        if x1 < 0:
            x1 = 1
        if y1 < 0:
            y1 = 1
        mask = np.zeros_like(original_frame)
        cv2.circle(mask, (x_o,y_o), 200, (255, 255, 255), -1, cv2.LINE_AA)
        # Apply the mask to the original image using bitwise operations
        masked_img = cv2.bitwise_and(original_frame, mask)
        if len(avg_center_x) <= 1:
            avg_center_x.append(center[0])

        if abs(center[0] - np.mean(avg_center_x)) >= 100 and not ball_move:
            print("Ball shot")
            count_shot += 1
            avg_center_x.clear()
            ball_move = True
        elif abs(center[0] - avg_center_x[0]) <= 10 and len(avg_center_x) != 5:
            avg_center_x.append(center[0])
        elif len(avg_center_x) < 5 and ball_move:
            avg_center_x.clear()
        elif len(avg_center_x) == 5:
            ball_move = False
        # Crop around white ball
        whiteball_zone = masked_img[y1:y2, x1:x2]
        

        # Cue Detection
        # Convert the frame to HSV color space
        #print(whiteball_zone.shape[:2])
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




    detectedBall = []
    detectedBallPos = []
    detectedBallTablePos = [] 




    cv2.namedWindow('Frame',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Frame", projection_frame)
    cv2.imshow("Frame2", whiteball_zone)
    cv2.imshow("Frame3", original_frame)



    #print('End Round')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()