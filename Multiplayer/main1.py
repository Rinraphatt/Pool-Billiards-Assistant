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

def showLine(frame, start, end , width = 10):
    if isShowline is True :
        cv2.line(frame, start, end, (255, 255, 255), width)

def realPosition(x,y):
    circle_pos_on_img = (int(roi_x + x), int(roi_y + y))

    homogeneous_coord = np.array([circle_pos_on_img[0], circle_pos_on_img[1], 1]).reshape(-1, 1)
    original_coord = np.matmul(M_inv, homogeneous_coord)
    original_coord /= original_coord[2]
    # The resulting original coordinate is (x_o, y_o)
    x_o = int(original_coord[0][0])
    y_o = int(original_coord[1][0])

    return x_o,y_o

def filter_ctrs(ctrs, min_s = 20, max_s = 500000, alpha = 1):  
    
    filtered_ctrs = [] # list for filtered contours
    
    for x in range(len(ctrs)): # for all contours
        
        rot_rect = cv2.minAreaRect(ctrs[x]) # area of rectangle around contour
        w = rot_rect[1][0] # width of rectangle
        h = rot_rect[1][1] # height
        area = cv2.contourArea(ctrs[x]) # contour area 

        
        if (h*alpha<w) or (w*alpha<h): # if the contour isnt the size of a snooker ball
            continue # do nothing
            
        if (area < min_s) or (area > max_s): # if the contour area is too big/small
            print("pif")
            continue # do nothing 

        # if it failed previous statements then it is most likely a ball
        filtered_ctrs.append(ctrs[x]) # add contour to filtered cntrs list

        
    return filtered_ctrs # returns filtere contours

def find_ctrs_color(ctrs, input_img):

    K = np.ones((3,3),np.uint8) # filter
    output = input_img.copy() #np.zeros(input_img.shape,np.uint8) # empty img
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) # gray version
    mask = np.zeros(gray.shape,np.uint8) # empty mask

    for i in range(len(ctrs)): # for all contours
        
        # find center of contour
        M = cv2.moments(ctrs[i])
        cX = int(M['m10']/M['m00']) # X pos of contour center
        cY = int(M['m01']/M['m00']) # Y pos
    
        mask[...]=0 # reset the mask for every ball 
    
        cv2.drawContours(mask,ctrs,i,255,-1) # draws the mask of current contour (every ball is getting masked each iteration)

        mask =  cv2.erode(mask,K,iterations=3) # erode mask to filter green color around the balls contours
        
        output = cv2.circle(output, # img to draw on
                         (cX,cY), # position on img
                         20, # radius of circle - size of drawn snooker ball
                         cv2.mean(input_img,mask), # color mean of each contour-color of each ball (src_img=transformed img)
                         -1) # -1 to fill ball with color
    return output

def draw_rectangles(ctrs, img):
    
    output = img.copy()
    
    for i in range(len(ctrs)):
    
        M = cv2.moments(ctrs[i]) # moments
        rot_rect = cv2.minAreaRect(ctrs[i])
        w = rot_rect[1][0] # width
        h = rot_rect[1][1] # height
        
        box = np.int64(cv2.boxPoints(rot_rect))
        cv2.drawContours(output,[box],0,(255,100,0),2) # draws box
        
    return output


def createTable():
    h,w = 880,1920
    frame = np.zeros((h, w, 3), np.uint8)
    pocket_point = [(0,0),(int(w/2),0),(w,0),(0,h),(int(w/2),h),(w,h)]
    for i in pocket_point:
        cv2.circle(frame, i , 50, (255, 255, 255), 5)
    return frame

def are_rectangles_overlapping(rect1, rect2):
    """
    Checks if two rectangles are overlapping
    rect1, rect2: tuples of 4 integers representing (x, y, width, height) of each rectangle
    returns: True if the rectangles are overlapping, False otherwise
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True

def circleOverlap(x1, y1, r1, x2, y2, r2):
    """
    Determines if a small circle with center (x1, y1) and radius r1 is inside a big circle with center (x2, y2) and radius r2.
    Returns True if the small circle is inside the big circle, False otherwise.
    """
    # Calculate the distance between the centers of the two circles
    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Determine if the small circle is inside the big circle or not
    if d + r1 < r2:
        return True
    elif d >= r2 - r1:
        return False
    else:
        return False  # The circles are tangent, so it's up to you to decide if the small circle is considered inside or not

# # White open light
# lowerColor = [
#     np.array([25,80,80]),   #Yellow
#     np.array([100,125,90]),  #Blue
#     np.array([0,25,220]),   #Red
#     np.array([125,90,130]), #Purple
#     np.array([11,100,129]), #Orange
#     np.array([75,70,90]),  #Green
#     np.array([140,70,30]),    #Crimson
#     np.array([55,0,0]),     #Black
#     np.array([0,0,180]),     #White

# ]

# upperColor = [
#     np.array([40,255,255]),  #Yellow
#     np.array([125,255,255]), #Blue
#     np.array([10,255,255]),  #Red
#     np.array([135,255,255]), #Purple
#     np.array([22,255,255]),  #Orange
#     np.array([95,255,255]),  #Green
#     np.array([179,255,255]),  #Crimson
#     np.array([179,145,100]),  #Black
#     np.array([179,60,255]), #White
# ]

# white close light 10:AM
lowerColor = [
    np.array([12,100,10]), #Yellow
    np.array([105,200,0]), #Blue
    np.array([150,150,160]), #Red
    np.array([128,170,0]), #Purple
    np.array([0,50,60]), #Orange
    np.array([85,155,85]), #Green
    np.array([135,170,80]), #Crimson
    np.array([100,0,0]), #Black
    np.array([120,0,170]), #White

]
upperColor = [
    np.array([35,255,255]),
    np.array([125,255,255]),
    np.array([179,255,255]),
    np.array([140,255,255]),
    np.array([10,255,255]),
    np.array([110,255,255]),
    np.array([170,255,180]),
    np.array([170,135,120]),
    np.array([179,120,255]),
]


# Load the camera matrix and distortion coefficients from the calibration file
mtx = np.loadtxt('../arUco/calib_data/camera_matrix.txt')
dist = np.loadtxt('../arUco/calib_data/dist_coeffs.txt')
cap = cv2.VideoCapture(0)
width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fps = cap.get(cv2.CAP_PROP_FPS)
print('fps = ', fps)
tl = (251 ,10)
bl = (183 ,908)
tr = (1709 ,27)
br = (1772 ,934)
table_width = 1920
table_height = 880
roi_x, roi_y = 0, 200
roi_w, roi_h = 1920, 880
isShowline = True
avg_center_x = []
avg_center_y = []
count_shot_p1 = 0
count_shot_p2 = 0
isP1 = True
ball_move = False

detectedBall = []
detectedBallPos = []
detectedBallTablePos = []

list_white = []
list_black = []
avg_white = (0,0)
avg_black = (0,0)
realtime_black = []

ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
pocket_point = [(240,166),(974,148),(1720,188),(180,905),(965,950),(1761,928)]
bound = 0
output_width = 1920 - bound
output_height = 880 - bound
output_min = 0 + bound
while True:
    succuess, frame = cap.read()
    whiteball_zone = np.zeros((400, 400, 3), np.uint8)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    # Load the image to be projected
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    original_frame = frame.copy()
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    M_inv = cv2.invert(M)[1]
    # Compute the perspective transform 
    for i in pocket_point:
        cv2.circle(frame, i , 50, (255, 255, 255), 5)
    transformed_frame = cv2.warpPerspective(frame, M, (width, height))
    projection_frame = transformed_frame.copy()
    table_frame = transformed_frame[200:1080,0:1920]
    
    black = createTable()
    # White close light 10 AM
    lower_green = np.array([50,0,0])
    upper_green = np.array([90,255,255])

    hsv = cv2.cvtColor(table_frame, cv2.COLOR_BGR2HSV)
    blurFrame = cv2.GaussianBlur(hsv, (7, 7), 0)
    mask = cv2.inRange(blurFrame, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # dilate->erode
    mask = cv2.dilate(mask_closing,kernel,iterations = 1)
    inv_mask = cv2.bitwise_not(mask)
    output = cv2.bitwise_and(table_frame,table_frame, mask= inv_mask)

    circles = cv2.HoughCircles(inv_mask, cv2.HOUGH_GRADIENT, 1, 30, param1=100, param2=10, minRadius=21, maxRadius=35)
    circleZones = []
    circleZonesColor = []
    if circles is not None :
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) <= 20 :
            i = 0
            for (x, y, r) in circles:
                x1 = x - 35
                y1 = y - 35
                x2 = x + 35
                y2 = y + 35
                if x1 < 0:
                    x1 = 1
                if y1 < 0:
                    y1 = 1
                circleZoneColor = table_frame[y1:y2, x1:x2]
                circleZonesColor.append(circleZoneColor)
                # i+=1
                # cv2.imshow("test"+str(i),circleZoneColor)
                cv2.circle(table_frame, (x,y), r, (0,255,0), 2)

    if circles is not None:
        # Find Color and Type of Every Balls
        for i in range(len(circleZonesColor)):
            maxSameColor = 0
            maxSameColorPos = -1
            semiSameColorPos = -1
            colorCounter = 0
            whiteCounter = 0

            if circleZonesColor[i].size != 0:
                hsvcircleZone = cv2.cvtColor(circleZonesColor[i], cv2.COLOR_BGR2HSV)
                for j in range(len(lowerColor)):
                    mask = cv2.inRange(hsvcircleZone, lowerColor[j], upperColor[j])
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # dilate->erode
                    mask = cv2.dilate(mask_closing,kernel,iterations = 1)
                    samePixs = np.sum(mask == 255)

                    if j == 8:
                        whiteCounter = samePixs

                    if samePixs > maxSameColor:
                        semiSameColorPos = maxSameColorPos
                        maxSameColor = samePixs
                        maxSameColorPos = j


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
                if (similarColor in ["Black","White"]):
                    # cv2.putText(table_frame, f'Number : {i}', (circles[i][0], circles[i][1]-80), cv2.FONT_HERSHEY_SIMPLEX, 
                    #     0.7, (255, 0, 255), 2, cv2.LINE_AA)
                    # cv2.putText(table_frame, f'Color : {similarColor}', (circles[i][0], circles[i][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 
                    #     0.7, (255, 0, 255), 2, cv2.LINE_AA)
                    # cv2.putText(table_frame, f'X : {circles[i][0]}', (circles[i][0], circles[i][1]-30), cv2.FONT_HERSHEY_SIMPLEX, 
                    #     0.7, (255, 0, 255), 2, cv2.LINE_AA)
                    # cv2.putText(table_frame, f'Y : {circles[i][1]}', (circles[i][0], circles[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    #     0.7, (255, 0, 255), 2, cv2.LINE_AA)
                    if similarColor == "Black":
                        realtime_black = [circles[i][0], circles[i][1],circles[i][2]]
                        if len(list_black) < 1:
                            list_black.append([circles[i][0], circles[i][1]])
                        # Define the points as numpy arrays
                        points = np.array(list_black)
                        # Calculate the mean of the points
                        avg_point = np.mean(points, axis=0)
                        ax,ay = avg_point
                        if abs(circles[i][0] - ax) < 10 and abs(circles[i][1] - ay) < 10 :
                            if len(list_black) < 3:
                                list_black.append([circles[i][0], circles[i][1]])
                            else :
                                avg_black = (round(ax),round(ay))
                        else:
                            list_black.pop(0)
                        

                    if similarColor == "White":
                        if len(list_white) < 1:
                            list_white.append([circles[i][0], circles[i][1]])
                        # Define the points as numpy arrays
                        points = np.array(list_white)
                        # Calculate the mean of the points
                        avg_point = np.mean(points, axis=0)
                        ax,ay = avg_point
                        if abs(circles[i][0] - ax) < 10 and abs(circles[i][1] - ay) < 10 :
                            if len(list_white) < 3:
                                list_white.append([circles[i][0], circles[i][1]])
                            else :
                                avg_white = (round(ax),round(ay))
                        else:
                            list_white.pop(0)

            for p in pocket_point:
                if len(realtime_black) != 0:
                    if circleOverlap(realtime_black[0], realtime_black[1], realtime_black[2], p[0], p[1], 50) :
                        print("End Round")
                   

    whitePos = -1
    if 'White' in detectedBall:
        whitePos = detectedBall.index('White')

    if whitePos != -1 :
        center = avg_white
        # Draw on black frame
        cv2.circle(black, center, 30, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.circle(black, center, 200, (255, 255, 255), 6, cv2.LINE_AA)

        x_o,y_o = realPosition(center[0],center[1])
        mask = np.zeros_like(original_frame)
        cv2.circle(mask, (x_o,y_o), 200, (255, 255, 255), -1, cv2.LINE_AA)
        masked_img = cv2.bitwise_and(original_frame, mask)
        if len(avg_center_x) <= 1:
            avg_center_x.append(center[0])
            avg_center_y.append(center[1])
        if (abs(center[0] - np.mean(avg_center_x)) >= 100 or abs(center[1] - np.mean(avg_center_y)) >= 100) and not ball_move:
            if isP1:
                count_shot_p1 += 1
                print("Ball shot for Player1")
            else:
                count_shot_p2 += 1
                print("Ball shot for Player2")
            avg_center_x.clear()
            avg_center_y.clear()
            ball_move = True
        elif abs(center[0] - avg_center_x[0]) <= 10 and len(avg_center_x) != 5:
            avg_center_x.append(center[0])
            avg_center_y.append(center[1])
        elif len(avg_center_x) < 5 and ball_move:
            avg_center_x.clear()
            avg_center_y.clear()
        elif len(avg_center_x) == 5:
            ball_move = False
        x1 = x_o - 200
        y1 = y_o - 200
        x2 = x_o + 200
        y2 = y_o + 200
        if x1 < 0:
            x1 = 1
        if y1 < 0:
            y1 = 1

        # Crop around white ball
        whiteball_zone = masked_img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(whiteball_zone, cv2.COLOR_BGR2HSV)
        blurFrame = cv2.GaussianBlur(hsv, (7, 7), 0)
        # Define a cue white color threshold
        lower_cue = np.array([145, 120, 140])
        upper_cue = np.array([170, 255, 255])
        mask = cv2.inRange(blurFrame, lower_cue, upper_cue)
        kernel = np.ones((3,3),np.uint8)
        mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # dilate->erode
        mask = cv2.dilate(mask_closing,kernel,iterations = 1)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Extract the largest contour that has more than 10 points
            contours = [c for c in contours if len(c) > 10]
            if not contours:
                continue
            else:
                cue_contour = max(contours, key=cv2.contourArea)
            cue_contour += (center[0]-200,center[1]-200)

            # Fit a line to the cue contour
            [vx, vy, x, y] = cv2.fitLine(cue_contour, cv2.DIST_L2, 0, 0.01, 0.01)

            # Compute the start and end points of the cue line
            start_x = int(x - vx * 2000)
            start_y = int(y - vy * 2000)
            end_x = int(x + vx * 2000)
            end_y = int(y + vy * 2000)


            
            height_cue, width_cue, _ = black.shape
            if start_x < 0:
                start_x = 0
                start_y = int(y + (start_x - x) * vy / vx)
            elif start_x >= width_cue:
                start_x = width_cue 
                start_y = int(y + (start_x - x) * vy / vx)
            if start_y < 0:
                start_y = 0
                start_x = int(x + (start_y - y) * vx / vy)
            elif start_y >= height_cue:
                start_y = height_cue 
                start_x = int(x + (start_y - y) * vx / vy)

            if end_x < 0:
                end_x = 0
                end_y = int(y + (end_x - x) * vy / vx)
            elif end_x >= width_cue:
                end_x = width_cue 
                end_y = int(y + (end_x - x) * vy / vx)
            if end_y < 0:
                end_y = 0
                end_x = int(x + (end_y - y) * vx / vy)
            elif end_y >= height_cue:
                end_y = height_cue 
                end_x = int(x + (end_y - y) * vx / vy)
            # Draw the cue line on the original frame

            # print("Top : "+ str(start_x) + "  "+str(start_y))
            # print("Bot : "+ str( end_x ) + "  " +str(end_y))
            cv2.line(black, (start_x, start_y), (end_x, end_y), (255, 255, 255), 10)
            
            output = cv2.bitwise_and(whiteball_zone, whiteball_zone, mask=mask)
            edges = cv2.Canny(output, 180, 255)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 45,minLineLength=10, maxLineGap=100)
            if lines is not None:
                if len(lines) >= 2:
                    x1, y1, x2, y2 = lines[0][0]
                    x3, y3, x4, y4 = lines[1][0]
                    # Point of line
                    # cv2.circle(black, (x1+center[0]-200, y1+center[1]-200), 2, (0, 255, 255), -1)
                    # cv2.circle(black, (x2+center[0]-200, y2+center[1]-200), 2, (0, 255, 255), -1)
                    # cv2.circle(black, (x3+center[0]-200, y3+center[1]-200), 2, (0, 255, 255), -1)
                    # cv2.circle(black, (x4+center[0]-200, y4+center[1]-200), 2, (0, 255, 255), -1)

                    m1 = (round((x1+x3) / 2), round((y1+y3) / 2))
                    m3 = (round((x2+x4) / 2), round((y2+y4) / 2))
                    # Convert point in Crop into Original frame
                    m1 = (m1[0]+center[0]-200, m1[1]+center[1]-200)
                    m3 = (m3[0]+center[0]-200, m3[1]+center[1]-200)
                    dis1ToCen = math.sqrt(pow(m1[0]-center[0], 2)+pow(m1[1]-center[1], 2))
                    dis2ToCen = math.sqrt(pow(m3[0]-center[0], 2)+pow(m3[1]-center[1], 2))
                    hor = ""
                    ver = ""
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

                    
                    x3, y3 = (0, 0)
                    x4, y4 = (0, 0)
                    # cv2.putText(black, "Ver : " + ver + ' Hor : ' + hor , (100, 300),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    if (ver == 'up' and hor == 'left'):
                        x_top, y_top = (start_x, start_y)
                        x_bottom, y_bottom = (end_x, end_y)
                        x3, y3 = (x_top, y_top)
                        if (y_top == output_min and (y_bottom == output_height or x_bottom == output_width)):
                            showLine(black,(x_top, y_top),(x_top-abs(x_bottom-x_top), output_height))
                            x4, y4 = (x_top-abs(x_bottom-x_top), output_height)

                        elif (x_top == output_min and (x_bottom == output_width or y_bottom == output_height)):
                            showLine(black,(x_top, y_top), (output_width,y_top-abs(y_top-y_bottom)))
                            print( (output_width,y_top-abs(y_top-y_bottom)))
                            x4, y4 = (output_width, y_top-abs(y_top-y_bottom))

                    if (ver == 'up' and hor == 'right'):
                        x_top, y_top = (end_x, end_y)
                        x_bottom, y_bottom = (start_x, start_y)
                        x3, y3 = (x_top, y_top)
                        if (y_top == output_min and (y_bottom == output_height or x_bottom == output_min)):
                            showLine(black,(x_top, y_top), (x_top+abs(x_bottom-x_top), output_height))
                            x4, y4 = (x_top+abs(x_bottom-x_top), output_height)

                        elif (x_top == output_width and (x_bottom == output_min or y_bottom == output_height)):
                            showLine(black,(x_top, y_top), (output_min,y_top-abs(y_top-y_bottom)))
                            x4, y4 = (output_min, y_top-abs(y_top-y_bottom))

                    if (ver == 'down' and hor == 'left'):
                        x_top, y_top = (end_x, end_y)
                        x_bottom, y_bottom = (start_x, start_y)
                        x3, y3 = (x_bottom, y_bottom)
                        if ((y_top == output_min or x_top == output_width) and y_bottom == output_height):
                            showLine(black,(x_bottom, y_bottom), (x_bottom-abs(x_bottom-x_top), 0))
                            x4, y4 = (x_bottom-abs(x_bottom-x_top), output_min)

                        elif ((x_top == output_width or y_top == output_min) and x_bottom == output_min):
                            showLine(black,(x_bottom, y_bottom), (output_width,y_bottom+abs(y_top-y_bottom)))
                            x4, y4 = (output_width, y_bottom +abs(y_top-y_bottom))
                    if (ver == 'down' and hor == 'right'):
                        x_top, y_top = (start_x, start_y)
                        x_bottom, y_bottom = (end_x, end_y)
                        x3, y3 = (x_bottom, y_bottom)
                        if ((y_top == output_min or x_top == output_min) and y_bottom == output_height):
                            showLine(black,(x_bottom, y_bottom), (x_bottom+abs(x_bottom-x_top), 0))
                            x4, y4 = (x_bottom+abs(x_bottom-x_top), output_min)

                        elif ((x_top == output_min or y_top == output_min) and x_bottom == output_width):
                            showLine(black,(x_bottom, y_bottom), (output_min,y_bottom+abs(y_top-y_bottom)))
                            x4, y4 = (0, y_bottom+abs(y_top-y_bottom))

                    if x3 >= x4:
                        hor = 'left'
                        if y3 >= y4:
                            ver = 'up'
                        else:
                            ver = 'down'
                    else:
                        hor = 'right'
                        if y3 >= y4:
                            ver = 'up'
                        else:
                            ver = 'down'

                    cv2.putText(black, "Ver : " + ver + ' Hor : ' + hor , (100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # Create Third reflex line
                    if (ver == 'up' and hor == 'left'):
                        if (y3 == output_height and y4 == output_min):
                            if x4 < output_min:
                                showLine(black,(output_min, int(findSlope(x3, y3, x4, y4, "y", output_min)[2])), (abs(x4), 0))

                            else:
                                showLine(black,(x4, y4), (x4-abs(x4-x3), output_height))    

                        elif (x3 == output_width and x4 == output_min):
                            if y4 < output_min:
                                showLine(black,(int(findSlope(x3, y3, x4, y4, "x", output_min)[2]), output_min), (0, abs(y4)))

                            else:
                                showLine(black,(x4, y4), (output_width, y4-abs(y4-y3)))

                    if (ver == 'up' and hor == 'right'):
                        if (y3 == output_height and y4 == output_min):
                            
                            if x4 > output_width:
                                showLine(black,(output_width, int(findSlope(x3, y3, x4, y4, "y", output_width)[2])), (output_width-abs(x4-output_width), output_min))

                            else:
                                showLine(black,(x4, y4), (x4+abs(x3-x4), output_height))

                        elif (x3 == output_min and x4 == output_width):
                            if y4 < output_min:
                                showLine(black,(int(findSlope(x3, y3, x4, y4, "x", output_min)[2]), output_min), (output_width, abs(y4)))

                            else:
                                showLine(black,(x4, y4), (output_min,y4-abs(y4-y3)))

                    if (ver == 'down' and hor == 'left'):
                        if (y3 == output_min and y4 == output_height):
                            if x4 < output_min:
                                showLine(black,(output_min, int(findSlope(x3, y3, x4, y4, "y", output_min)[2])), (abs(x4), output_height))

                            else:
                                showLine(black,(x4, y4), (x4-abs(x4-x3), output_min))

                        elif (x3 == output_width and x4 == output_min):
                            if y4 > output_height:
                                showLine(black,(int(findSlope(x3, y3, x4, y4, "x", output_height)[2]), output_height), (output_min, output_height-(y4-output_height)))
                            else:
                                showLine(black,(x4, y4), (output_width, y4-abs(y4-y3)))

                    if (ver == 'down' and hor == 'right'):
                        if (y3 == output_min and y4 == output_height):
                            if x4 > output_width:
                                showLine(black,(output_width, int(findSlope(x3, y3, x4, y4, "y", output_width)[2])), (output_width-abs(output_width-x4), output_height))

                            else:
                                showLine(black,(x4, y4), (x4+abs(x4-x3), output_min))

                        elif (x3 == output_min and x4 == output_width):
                            if y4 > output_height:
                                showLine(black,(int(findSlope(x3, y3, x4, y4, "x", output_height)[2]), output_height), (output_width, output_height-abs(y4-output_height)))

                            else:
                                showLine(black,(x4, y4), (output_min,y4+abs(y4-y3)))

    black_bg = np.zeros((1080, 1920, 3), np.uint8)
    black_bg[200:1080,0:1920] = black 
    cv2.namedWindow('Frame',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Frame", black_bg)
    cv2.imshow("Frame1", table_frame)
    #cv2.imshow("Frame2", black)
    #cv2.imshow("Frame3", whiteball_zone)





    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()