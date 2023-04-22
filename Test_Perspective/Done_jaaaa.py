import numpy as np
import cv2
import time

vidcap = cv2.VideoCapture(0)
width = 1920
height = 1080
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

# Define the lower and upper bounds of the skin color in the HSV color space
lower_skin = np.array([80, 200, 80], dtype=np.uint8)
upper_skin = np.array([110, 255, 255], dtype=np.uint8)

# Define the rectangular zone
x1NextBtn, y1NextBtn = 1750, 710  # top-left corner
x2NextBtn, y2NextBtn = 1850, 810  # bottom-right corner

x1PrevBtn, y1PrevBtn = 1600, 710  # top-left corner
x2PrevBtn, y2PrevBtn = 1700, 810  # bottom-right corner

# Initialize the timer
start_time = None

# stage1Pics = [
#     'stage1.png',
#     'stage1_2.png',
#     'stage1_3.png'
# ]

# stage1Pics = [
#     'stage2.png',
#     'stage2_2.png',
#     'stage2_3.png',
#     'stage2_4.png'
# ]

# stage1Pics = [
#     'modeBasic.png'
# ]

# stage1Pics = [
#     'stageDiamond.png',
#     'stageDiamond_2.png',
#     'stageDiamond_3.png',
#     'stageDiamond_4.png',
#     'stageDiamond_5.png',
#     'stageDiamond_6.png',
#     'stageDiamond_7.png',
#     'stageDiamond_8.png',
#     'stageDiamond_9.png',
# ]

# stage1Pics = [
#     'stageBallControl.png',
#     'stageBallControl_2.png',
#     'stageBallControl_3.png',
#     'stageBallControl_4.png',
#     'stageBallControl_5.png',
#     'stageBallControl_6.png',
#     'stageBallControl_7.png',
# ]

stage1State = 0

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

# Load the camera matrix and distortion coefficients from the calibration file
mtx = np.loadtxt('./arUco/calib_data/camera_matrix.txt')
dist = np.loadtxt('./arUco/calib_data/dist_coeffs.txt')
print("Loaded")
mac = cv2.imread('./pics/Stage/'+stage1Pics[0])
mac = cv2.resize(mac, (1920, 880))
while True:
    succuess, img = vidcap.read()
    frame = img
    frame = cv2.undistort(frame, mtx, dist)
    # tl = (252 ,21)
    # bl = (174 ,906)
    # tr = (1695 ,31)
    # br = (1748 ,933)
    tl = (245 ,10)
    bl = (180 ,900)
    tr = (1717 ,22)
    br = (1760 ,930)
    #cv2.circle(frame, tl, 3, (0, 0, 255), -1)
    # cv2.circle(frame, bl, 3, (0, 0, 255), -1)
    # cv2.circle(frame, tr, 3, (0, 0, 255), -1)
    # cv2.circle(frame, br, 3, (0, 0, 255), -1)
    # cv2.line(frame, tl, bl, (0, 255, 0), 2)
    # cv2.line(frame, bl, br, (0, 255, 0), 2)
    # cv2.line(frame, br, tr, (0, 255, 0), 2)
    # cv2.line(frame, tl, tr, (0, 255, 0), 2)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    tansformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
    frame2 = tansformed_frame.copy()
    handDetectFrame = frame2[200:1080,0:1920]
    tansformed_frame[200:1080,0:1920] = mac

    hsv = cv2.cvtColor(handDetectFrame, cv2.COLOR_BGR2HSV)

    # Apply the skin color segmentation to the HSV frame
    maskHand = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to remove noise and fill holes in the mask
    kernel = np.ones((2, 2), np.uint8)
    maskHand = cv2.erode(maskHand, kernel, iterations=1)
    maskHand = cv2.dilate(maskHand, kernel, iterations=1)

    cv2.imshow('Hand Zone1', maskHand)

    # Find the contours in the mask
    contours, hierarchy = cv2.findContours(maskHand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw a rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(handDetectFrame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(tansformed_frame, (x, y+200), (x+w, y+h+200), (0, 255, 0), 2)
        rect1 = x,y,w,h
        rect2 = x1NextBtn,y1NextBtn,x2NextBtn-x1NextBtn,y2NextBtn-y1NextBtn
        rect3 = x1PrevBtn,y1PrevBtn,x2PrevBtn-x1PrevBtn,y2PrevBtn-y1PrevBtn
        # Draw the rectangular zone on the frame
        cv2.rectangle(handDetectFrame, (x1NextBtn, y1NextBtn), (x2NextBtn, y2NextBtn), (0, 255, 0), 3)
        cv2.rectangle(tansformed_frame, (x1NextBtn, y1NextBtn+200), (x2NextBtn, y2NextBtn+200), (0, 255, 0), 3)
        cv2.rectangle(handDetectFrame, (x1PrevBtn, y1PrevBtn), (x2PrevBtn, y2PrevBtn), (255, 0, 0), 3)
        cv2.rectangle(tansformed_frame, (x1PrevBtn, y1PrevBtn+200), (x2PrevBtn, y2PrevBtn+200), (255, 0, 0), 3)
        if are_rectangles_overlapping(rect1,rect2) == True : 
            if start_time is None:
                start_time = time.time()

            elif time.time() - start_time >= 2:
                print('Hand stayed in the zone for 3 seconds!')
                cv2.rectangle(handDetectFrame, (x1NextBtn, y1NextBtn), (x2NextBtn, y2NextBtn), (0, 0, 255), 3)
                cv2.rectangle(tansformed_frame, (x1NextBtn, y1NextBtn+200), (x2NextBtn, y2NextBtn+200), (0, 0, 255), 3)
                stage1State += 1
                if stage1State >= len(stage1Pics) :
                    stage1State = len(stage1Pics) - 1
                    
                print('Stage1State : ', stage1State)
                mac = cv2.imread('./pics/Stage/'+stage1Pics[stage1State])
                mac = cv2.resize(mac, (1920, 880))
                start_time = time.time()
        elif are_rectangles_overlapping(rect1,rect3) == True :
            if start_time is None:
                start_time = time.time()

            elif time.time() - start_time >= 2:
                print('Hand stayed in the zone for 3 seconds!')
                cv2.rectangle(handDetectFrame, (x1PrevBtn, y1PrevBtn), (x2PrevBtn, y2PrevBtn), (0, 0, 255), 3)
                cv2.rectangle(tansformed_frame, (x1PrevBtn, y1PrevBtn+200), (x2PrevBtn, y2PrevBtn+200), (0, 0, 255), 3)
                stage1State -= 1
                if stage1State <= 0 :
                    stage1State = 0
                    
                print('Stage1State : ', stage1State)
                mac = cv2.imread('./pics/Stage/'+stage1Pics[stage1State])
                mac = cv2.resize(mac, (1920, 880))
                start_time = time.time()
        else:
            start_time = None
  
    cv2.namedWindow('Test_Perspectice',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Test_Perspectice', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Test_Perspectice", tansformed_frame)
    # cv2.imshow('Hand Zone', handDetectFrame)
    #cv2.imshow("Test", frame)

    
    if cv2.waitKey(1) == 27:
        break
