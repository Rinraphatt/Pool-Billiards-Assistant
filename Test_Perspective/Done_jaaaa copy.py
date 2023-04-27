import numpy as np
import cv2
import time
import random
import math
import BallDetectionLib as BDLib
import sys
sys.path.append('./BallDetection/')
import BallDetectionLibtest as BDLib1

vidcap = cv2.VideoCapture(0)
width = 1920
height = 1080
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
prevBallPos = []
# Define the lower and upper bounds of the skin color in the HSV color space [NightBlue]
lower_skin = np.array([80, 70, 70], dtype=np.uint8)
upper_skin = np.array([100, 255, 255], dtype=np.uint8)

# Define the lower and upper bounds of the skin color in the HSV color space [NightBlue]
lower_skin_white = np.array([70, 0, 100], dtype=np.uint8)
upper_skin_white = np.array([85, 255, 255], dtype=np.uint8)

# Define the lower and upper bounds of the skin color in the HSV color space [White Light]
lower_white_light = np.array([10, 0, 0], dtype=np.uint8)
upper_white_light = np.array([40, 90, 255], dtype=np.uint8)

# Define the lower and upper bounds of the skin color in the HSV color space [Morning]
# lower_skin = np.array([80, 30, 180], dtype=np.uint8)
# upper_skin = np.array([100, 255, 255], dtype=np.uint8)

# Define the lower and upper bounds of the skin color in the HSV color space [Noon]
# lower_skin = np.array([60, 20, 200], dtype=np.uint8)
# upper_skin = np.array([100, 255, 255], dtype=np.uint8)

# Define the rectangular zone
x1NextBtn, y1NextBtn = 1750, 710  # top-left corner
x2NextBtn, y2NextBtn = 1850, 810  # bottom-right corner

x1PrevBtn, y1PrevBtn = 1600, 710  # top-left corner
x2PrevBtn, y2PrevBtn = 1700, 810  # bottom-right corner

x1BackBtn, y1BackBtn = 1750, 70  # top-left corner
x2BackBtn, y2BackBtn = 1850, 170  # bottom-right corner

x1TrainingBtn, y1TrainingBtn = 611, 105  # top-left corner
x2TrainingBtn, y2TrainingBtn = 1310, 403  # bottom-right corner

x1GamePlayBtn, y1GamePlayBtn = 611, 477  # top-left corner
x2GamePlayBtn, y2GamePlayBtn = 1310, 775  # bottom-right corner

x1BasicBtn, y1BasicBtn = 234, 394  # top-left corner
x2BasicBtn, y2BasicBtn = 681, 763  # bottom-right corner

x1AmatureBtn, y1AmatureBtn = 737, 394  # top-left corner
x2AmatureBtn, y2AmatureBtn = 1184, 763  # bottom-right corner

x1ProBtn, y1ProBtn = 1239, 394  # top-left corner
x2ProBtn, y2ProBtn = 1686, 763  # bottom-right corner

x1SingleBtn, y1SingleBtn = 259, 252  # top-left corner
x2SingleBtn, y2SingleBtn = 895, 791  # bottom-right corner

x1MultiBtn, y1MultiBtn = 1026, 252  # top-left corner
x2MultiBtn, y2MultiBtn = 1662, 791  # bottom-right corner

x1ReplayBtn, y1ReplayBtn = 271, 702  # top-left corner
x2ReplayBtn, y2ReplayBtn = 472, 764  # bottom-right corner

x1ExitBtn, y1ExitBtn = 1441, 702  # top-left corner
x2ExitBtn, y2ExitBtn = 1642, 764  # bottom-right corner

# white close light 3:AM
lowerColor1 = [
    np.array([12,100,120]), #Yellow
    np.array([115,200,0]), #Blue
    np.array([150,150,180]), #Red
    np.array([128,215,0]), #Purple
    np.array([0,75,150]), #Orange
    np.array([100,155,110]), #Green
    np.array([150,180,75]), #Crimson
    np.array([100,0,0]), #Black
    np.array([120,0,200]), #White

]
upperColor1 = [
    np.array([35,255,255]),
    np.array([125,255,255]),
    np.array([179,255,255]),
    np.array([160,255,255]),
    np.array([10,255,255]),
    np.array([110,255,255]),
    np.array([170,255,175]),
    np.array([170,135,90]),
    np.array([179,105,255]),
]
# Initialize the timer
start_time = None
debounceTime = 1.5

stageBasic1Pics = [
    'stageBridge.png',
    'stageBridge_2.png',
    'stageBridge_3.png',
    'stageBridge_4.png',
]

stageBasic2Pics = [
    'stageStraight.png',
    'stageStraight_2.png',
    'stageStraight_3.png'
]

stageBasic3Pics = [
    'stageGhost.png',
    'stageGhost_2.png',
    'stageGhost_3.png',
    'stageGhost_4.png'
]

stageAmature1Pics = [
    'stageCombination.png',
    'stageCombination_2.png',
    'stageCombination_3.png',
    'stageCombination_4.png',
]

stageAmature2Pics = [
    'stageDiamond.png',
    'stageDiamond_2.png',
    'stageDiamond_3.png',
    'stageDiamond_4.png',
    'stageDiamond_5.png',
    'stageDiamond_6.png',
    'stageDiamond_7.png',
    'stageDiamond_8.png',
    'stageDiamond_9.png',
]

stageAmature3Pics = [
    'stageBallControl.png',
    'stageBallControl_2.png',
    'stageBallControl_3.png',
    'stageBallControl_4.png',
    'stageBallControl_5.png',
    'stageBallControl_6.png',
    'stageBallControl_7.png',
]

stagePro1Pics = [
    'stageBasicPositioning.png',
    'stageBasicPositioning_2.png',
    'stageBasicPositioning_3.png',
    'stageBasicPositioning_4.png',
    'stageBasicPositioning_5.png',
    'stageBasicPositioning_6.png',
]

stagePro2Pics = [
    'stageSideSpin.png',
    'stageSideSpin_2.png',
    'stageSideSpin_3.png',
    'stageSideSpin_4.png',
    'stageSideSpin_5.png',
    'stageSideSpin_6.png',
]

stagePro3Pics = [
    'stageMasse.png',
    'stageMasse_2.png',
    'stageMasse_3.png',
    'stageMasse_4.png',
    'stageMasse_5.png',
    'stageMasse_6.png',
]

currentStagePics = []

# stage1Pics = [
#     'stageGrid.png'
# ]

mainStage = 'modeSelect'
prevMainStage = ''
currentStageState = 0
timeTrialMaxTime = 180
timeTrialStartTime = 0
timeTrialState = 'Prepare'
timeTrialScore = 0
timeTrialHighScore = 0
ballCheckingStartTime = 0
timeTrialMainBall = 'Blue'
timeTrialObjBall = 'Red'

objectBallExist = True
objectX = random.randint(20, 1900)
objectY = random.randint(20, 780)

updatedBall = []
updatedBallPos = []
updatedBallTablePos = []
detectedBall = []
detectedBallPos = []
detectedBallTablePos = []

ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]

bgFrame = None

prevObjectX = 0
prevObjectY = 0

prevWhiteX = 0
prevWhiteY = 0

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
def showLine(frame, start, end , width = 5):
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
# Load the camera matrix and distortion coefficients from the calibration file
mtx = np.loadtxt('./arUco/calib_data/camera_matrix.txt')
dist = np.loadtxt('./arUco/calib_data/dist_coeffs.txt')
print("Loaded")
# mac = cv2.imread('./pics/Stage/modeSelect')
mac = cv2.imread('./pics/Stage/modeSelect.png')
mac = cv2.resize(mac, (1920, 880))
#bg = np.zeros((200, 1920, 3), np.uint8)
while True:
    succuess, frame = vidcap.read()
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    # Load the image to be projected
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # tl = (252 ,21)
    # bl = (174 ,906)
    # tr = (1695 ,31)
    # br = (1748 ,933)
    tl = (251 ,10)
    bl = (183 ,908)
    tr = (1709 ,27)
    br = (1772 ,934)
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
    M = cv2.getPerspectiveTransform(pts1, pts2)
    M_inv = cv2.invert(M)[1]
    # Compute the perspective transform M
    tansformed_frame = cv2.warpPerspective(frame, M, (width, height))
    frame2 = tansformed_frame.copy()
    frame3 = tansformed_frame.copy()
    original_frame = tansformed_frame.copy()
    handDetectFrame = frame2[200:1080,0:1920]
    table_frame = frame3[200:1080,0:1920]
    
    #tansformed_frame[0:200,0:1920] = bg
    tansformed_frame[200:1080,0:1920] = mac
    
    hsv = cv2.cvtColor(handDetectFrame, cv2.COLOR_BGR2HSV)

    # Apply the skin color segmentation to the HSV frame
    if mainStage == 'timeTrialMode' :
        maskHand = cv2.inRange(hsv, lower_skin_white, upper_skin_white)
    elif mainStage == 'multiplayerMode':
        maskHand = cv2.inRange(hsv, lower_white_light, upper_white_light)
    else :
        # maskHand = cv2.inRange(hsv, lower_skin, upper_skin)
        maskHand = cv2.inRange(hsv, lower_skin_white, upper_skin_white)

    # Apply morphological operations to remove noise and fill holes in the mask
    kernel = np.ones((2, 2), np.uint8)
    maskHand = cv2.erode(maskHand, kernel, iterations=1)
    maskHand = cv2.dilate(maskHand, kernel, iterations=1)

    # cv2.imshow('Hand Zone1', maskHand)

    # Find the contours in the mask
    contours, hierarchy = cv2.findContours(maskHand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw a rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # cv2.rectangle(handDetectFrame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(tansformed_frame, (x, y+200), (x+w, y+h+200), (0, 255, 0), 2)

        # Create Button Rectangle
        cursorRect = x,y,w,h
        nextBtnRect = x1NextBtn,y1NextBtn,x2NextBtn-x1NextBtn,y2NextBtn-y1NextBtn
        prevBtnRect = x1PrevBtn,y1PrevBtn,x2PrevBtn-x1PrevBtn,y2PrevBtn-y1PrevBtn
        backBtnRect = x1BackBtn,y1BackBtn,x2BackBtn-x1BackBtn,y2BackBtn-y1BackBtn
        trainingBtnRect = x1TrainingBtn,y1TrainingBtn,x2TrainingBtn-x1TrainingBtn,y2TrainingBtn-y1TrainingBtn
        gamePlayBtnRect = x1GamePlayBtn,y1GamePlayBtn,x2GamePlayBtn-x1GamePlayBtn,y2GamePlayBtn-y1GamePlayBtn
        basicBtnRect = x1BasicBtn,y1BasicBtn,x2BasicBtn-x1BasicBtn,y2BasicBtn-y1BasicBtn
        amatureBtnRect = x1AmatureBtn,y1AmatureBtn,x2AmatureBtn-x1AmatureBtn,y2AmatureBtn-y1AmatureBtn
        proBtnRect = x1ProBtn,y1ProBtn,x2ProBtn-x1ProBtn,y2ProBtn-y1ProBtn
        leftStageBtnRect = basicBtnRect
        middleStageBtnRect = amatureBtnRect
        rightStageBtnRect = proBtnRect
        singleBtnRect = x1SingleBtn,y1SingleBtn,x2SingleBtn-x1SingleBtn,y2SingleBtn-y1SingleBtn
        multiBtnRect = x1MultiBtn,y1MultiBtn,x2MultiBtn-x1MultiBtn,y2MultiBtn-y1MultiBtn
        replayBtnRect = x1ReplayBtn,y1ReplayBtn,x2ReplayBtn-x1ReplayBtn,y2ReplayBtn-y1ReplayBtn
        exitBtnRect = x1ExitBtn,y1ExitBtn,x2ExitBtn-x1ExitBtn,y2ExitBtn-y1ExitBtn

        # Draw the rectangular zone on the frame
        # cv2.rectangle(handDetectFrame, (x1NextBtn, y1NextBtn), (x2NextBtn, y2NextBtn), (0, 255, 0), 3)
        # cv2.rectangle(handDetectFrame, (x1PrevBtn, y1PrevBtn), (x2PrevBtn, y2PrevBtn), (255, 0, 0), 3)
        if mainStage == 'modeSelect' :
            cv2.rectangle(tansformed_frame, (x1TrainingBtn, y1TrainingBtn+200), (x2TrainingBtn, y2TrainingBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1GamePlayBtn, y1GamePlayBtn+200), (x2GamePlayBtn, y2GamePlayBtn+200), (255, 0, 0), 3)
        elif mainStage == 'difficultSelect' or mainStage == 'basicStageSelect' or mainStage == 'amatureStageSelect' or mainStage == 'proStageSelect':
            cv2.rectangle(tansformed_frame, (x1BasicBtn, y1BasicBtn+200), (x2BasicBtn, y2BasicBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1AmatureBtn, y1AmatureBtn+200), (x2AmatureBtn, y2AmatureBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1ProBtn, y1ProBtn+200), (x2ProBtn, y2ProBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 0, 255), 3)
        elif mainStage == 'trainingStage' :
            cv2.rectangle(tansformed_frame, (x1NextBtn, y1NextBtn+200), (x2NextBtn, y2NextBtn+200), (0, 255, 0), 3)
            cv2.rectangle(tansformed_frame, (x1PrevBtn, y1PrevBtn+200), (x2PrevBtn, y2PrevBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 0, 255), 3)
        elif mainStage == 'gamePlaySelect' or mainStage == 'singleplayerModeSelect':
            cv2.rectangle(tansformed_frame, (x1SingleBtn, y1SingleBtn+200), (x2SingleBtn, y2SingleBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1MultiBtn, y1MultiBtn+200), (x2MultiBtn, y2MultiBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 0, 255), 3)
        elif mainStage == 'freedomMode' or mainStage == 'timeTrialMode' or mainStage == 'multiplayerMode':
            cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 0, 255), 3)

        if mainStage == 'modeSelect': 
            if are_rectangles_overlapping(cursorRect,trainingBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'modeSelect'
                    mainStage = 'difficultSelect'
                    mac = cv2.imread('./pics/Stage/difficultSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,gamePlayBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'modeSelect'
                    mainStage = 'gamePlaySelect'
                    mac = cv2.imread('./pics/Stage/gamePlaySelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            else:
                start_time = None

        elif mainStage == 'difficultSelect':
            if are_rectangles_overlapping(cursorRect,basicBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'difficultSelect'
                    mainStage = 'basicStageSelect'
                    mac = cv2.imread('./pics/Stage/basicStageSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,amatureBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'difficultSelect'
                    mainStage = 'amatureStageSelect'
                    mac = cv2.imread('./pics/Stage/amatureStageSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,proBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'difficultSelect'
                    mainStage = 'proStageSelect'
                    mac = cv2.imread('./pics/Stage/proStageSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'modeSelect'
                    mainStage = 'modeSelect'
                    mac = cv2.imread('./pics/Stage/modeSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            else:
                start_time = None
        elif mainStage == 'gamePlaySelect':
            if are_rectangles_overlapping(cursorRect,singleBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'gamePlaySelect'
                    mainStage = 'singleplayerModeSelect'
                    mac = cv2.imread('./pics/Stage/singleplayerModeSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,multiBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'gamePlaySelect'
                    mainStage = 'multiplayerMode'
                    mac = BDLib1.createTable()
                    start_time = time.time()
                    roi_x, roi_y = 0,200
                    roi_w, roi_h = 1920, 880
                    isShowline = True
                    isDraw = True
                    avg_center_x = []
                    avg_center_y = []
                    count_shot_p1 = 1
                    count_shot_p2 = 1
                    isP1 = True
                    ball_move = False
                    avg_cue_x1 = []
                    avg_cue_y1 = []
                    avg_cue_x2 = []
                    avg_cue_y2 = []
                    list_start = []
                    list_end = []
                    detectedBall = []
                    detectedBallPos = []
                    detectedBallTablePos = []
                    updatedBall = []
                    updatedBallPos = []
                    updatedBallTablePos = []
                    ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    check_black = None
                    list_white = []
                    list_black = []
                    avg_white = (0,0)
                    avg_black = (0,0)
                    realtime_black = []
                    acurency_p1 = 0
                    acurency_p2 = 0
                    ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    pocket_point = [(240,166),(974,148),(1720,188),(180,905),(965,950),(1761,928)]
                    bound = 0
                    output_width = 1920 - bound
                    output_height = 880 - bound
                    output_min = 0 + bound

            elif are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'modeSelect'
                    mainStage = 'modeSelect'
                    mac = cv2.imread('./pics/Stage/modeSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            else:
                start_time = None
        elif mainStage == 'singleplayerModeSelect':
            if are_rectangles_overlapping(cursorRect,singleBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'singleplayerModeSelect'
                    mainStage = 'freedomMode'
                    mac = np.zeros((880, 1920, 3), np.uint8)
                    cv2.rectangle(mac, (0, 0), (1920, 880), (0, 0, 0), -1)
                    start_time = time.time()
                    roi_x, roi_y = 0, 200
                    roi_w, roi_h = 1920, 880
                    isShowline = True
                    isDraw = True
                    avg_center_x = []
                    avg_center_y = []
                    count_shot_p1 = 1
                    count_shot_p2 = 1
                    isP1 = True
                    ball_move = False
                    avg_cue_x1 = []
                    avg_cue_y1 = []
                    avg_cue_x2 = []
                    avg_cue_y2 = []
                    list_start = []
                    list_end = []
                    detectedBall = []
                    detectedBallPos = []
                    detectedBallTablePos = []
                    updatedBall = []
                    updatedBallPos = []
                    updatedBallTablePos = []
                    
                    ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    check_black = None
                    list_white = []
                    list_black = []
                    avg_white = (0,0)
                    avg_black = (0,0)
                    realtime_black = []
                    acurency_p1 = 0
                    acurency_p2 = 0
                    ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    pocket_point = [(240,166),(974,148),(1720,188),(180,905),(965,950),(1761,928)]
                    bound = 0
                    output_width = 1920 - bound
                    output_height = 880 - bound
                    output_min = 0 + bound


            elif are_rectangles_overlapping(cursorRect,multiBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'singleplayerModeSelect'
                    mainStage = 'timeTrialMode'
                    mac = np.zeros((880, 1920, 3), np.uint8)
                    cv2.rectangle(mac, (0, 0), (1920, 880), (0, 0, 0), -1)
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'modeSelect'
                    mainStage = 'gamePlaySelect'
                    mac = cv2.imread('./pics/Stage/gamePlaySelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            else:
                start_time = None
        elif mainStage == 'multiplayerMode':
            if are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'modeSelect'
                    mainStage = 'gamePlaySelect'
                    mac = cv2.imread('./pics/Stage/gamePlaySelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,nextBtnRect) == True: 
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1NextBtn, y1NextBtn+200), (x2NextBtn, y2NextBtn+200), (0, 0, 255), 3)
                    isP1 = not isP1
                    start_time = time.time()
            
            else:
                start_time = None
        elif mainStage == 'freedomMode':
            if are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'gamePlaySelect'
                    mainStage = 'singleplayerModeSelect'
                    mac = cv2.imread('./pics/Stage/singleplayerModeSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            else:
                start_time = None
        elif mainStage == 'timeTrialMode':
            if are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'gamePlaySelect'
                    mainStage = 'singleplayerModeSelect'
                    mac = cv2.imread('./pics/Stage/singleplayerModeSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    updatedBall = []
                    updatedBallPos = []
                    updatedBallTablePos = []
                    detectedBall = []
                    detectedBallPos = []
                    detectedBallTablePos = []
                    ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    bgFrame = None
                    prevObjectX = 0
                    prevObjectY = 0
                    prevWhiteX = 0
                    prevWhiteY = 0
                    timeTrialState = 'Prepare'
                    timeTrialScore = 0
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,replayBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    timeTrialState = 'Prepare'
                    updatedBall = []
                    updatedBallPos = []
                    updatedBallTablePos = []
                    detectedBall = []
                    detectedBallPos = []
                    detectedBallTablePos = []
                    ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    bgFrame = None
                    prevObjectX = 0
                    prevObjectY = 0
                    prevWhiteX = 0
                    prevWhiteY = 0
                    timeTrialScore = 0
                    ballCheckingStartTime = 0
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,exitBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    timeTrialState = 'Prepare'
                    prevMainStage = 'gamePlaySelect'
                    mainStage = 'singleplayerModeSelect'
                    mac = cv2.imread('./pics/Stage/singleplayerModeSelect.png')
                    mac = cv2.resize(mac, (1920, 880))
                    updatedBall = []
                    updatedBallPos = []
                    updatedBallTablePos = []
                    detectedBall = []
                    detectedBallPos = []
                    detectedBallTablePos = []
                    ballProbs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    bgFrame = None
                    prevObjectX = 0
                    prevObjectY = 0
                    prevWhiteX = 0
                    prevWhiteY = 0
                    timeTrialScore = 0
                    ballCheckingStartTime = 0
                    start_time = time.time()
            else:
                start_time = None
        elif mainStage == 'basicStageSelect':
            if are_rectangles_overlapping(cursorRect,leftStageBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'basicStageSelect'
                    mainStage = 'trainingStage'
                    currentStagePics = stageBasic1Pics
                    currentStageState = 0
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,middleStageBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'basicStageSelect'
                    mainStage = 'trainingStage'
                    currentStagePics = stageBasic2Pics
                    currentStageState = 0
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,rightStageBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'basicStageSelect'
                    mainStage = 'trainingStage'
                    currentStagePics = stageBasic3Pics
                    currentStageState = 0
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'modeSelect'
                    mainStage = 'difficultSelect'
                    mac = cv2.imread('./pics/Stage/difficultSelect.png') 
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            else:
                start_time = None
        elif mainStage == 'amatureStageSelect':
            if are_rectangles_overlapping(cursorRect,leftStageBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'amatureStageSelect'
                    mainStage = 'trainingStage'
                    currentStagePics = stageAmature1Pics
                    currentStageState = 0
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,middleStageBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'amatureStageSelect'
                    mainStage = 'trainingStage'
                    currentStagePics = stageAmature2Pics
                    currentStageState = 0
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,rightStageBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'amatureStageSelect'
                    mainStage = 'trainingStage'
                    currentStagePics = stageAmature3Pics
                    currentStageState = 0
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'modeSelect'
                    mainStage = 'difficultSelect'
                    mac = cv2.imread('./pics/Stage/difficultSelect.png') 
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            else:
                start_time = None
        elif mainStage == 'proStageSelect':
            if are_rectangles_overlapping(cursorRect,leftStageBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'proStageSelect'
                    mainStage = 'trainingStage'
                    currentStagePics = stagePro1Pics
                    currentStageState = 0
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,middleStageBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'proStageSelect'
                    mainStage = 'trainingStage'
                    currentStagePics = stagePro2Pics
                    currentStageState = 0
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,rightStageBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    prevMainStage = 'proStageSelect'
                    mainStage = 'trainingStage'
                    currentStagePics = stagePro3Pics
                    currentStageState = 0
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'modeSelect'
                    mainStage = 'difficultSelect'
                    mac = cv2.imread('./pics/Stage/difficultSelect.png') 
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            else:
                start_time = None
        elif mainStage == 'trainingStage' :
            if are_rectangles_overlapping(cursorRect,nextBtnRect) == True: 
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1NextBtn, y1NextBtn+200), (x2NextBtn, y2NextBtn+200), (0, 0, 255), 3)
                    currentStageState += 1
                    if currentStageState >= len(currentStagePics) :
                        currentStageState = len(currentStagePics) - 1
                        
                    print('currentStageState : ', currentStageState)
                    mac = cv2.imread('./pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,prevBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1PrevBtn, y1PrevBtn+200), (x2PrevBtn, y2PrevBtn+200), (0, 0, 255), 3)
                    currentStageState -= 1
                    if currentStageState <= 0 :
                        currentStageState = 0
                        
                    print('currentStageState : ', currentStageState)
                    mac = cv2.imread('./pics/Stage/'+currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    
                    if prevMainStage == 'basicStageSelect' : 
                        prevMainStage = 'difficultSelect'
                        mainStage = 'basicStageSelect'
                        mac = cv2.imread('./pics/Stage/basicStageSelect.png')
                    elif prevMainStage == 'amatureStageSelect' : 
                        prevMainStage = 'difficultSelect'
                        mainStage = 'amatureStageSelect'
                        mac = cv2.imread('./pics/Stage/amatureStageSelect.png')
                    elif prevMainStage == 'proStageSelect' : 
                        prevMainStage = 'difficultSelect'
                        mainStage = 'proStageSelect'
                        mac = cv2.imread('./pics/Stage/proStageSelect.png')

                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            else:
                start_time = None
  
    if mainStage == 'timeTrialMode':
        mac = np.zeros((880, 1920, 3), np.uint8)
        cv2.rectangle(mac, (0, 0), (1920, 880), (0, 0, 0), -1)

        # Check if there are motion or not
        motionGrayFrame = cv2.cvtColor(handDetectFrame, cv2.COLOR_BGR2GRAY)
        motionBlurFrame = cv2.GaussianBlur(motionGrayFrame, (35,35), 0)

        if bgFrame is None:
            bgFrame = motionBlurFrame

        frameDelta = cv2.absdiff(bgFrame, motionBlurFrame)
        _,thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)

        n_white_pix = np.sum(thresh == 255)

        if n_white_pix <= 100:
            res = BDLib.getCircles(handDetectFrame,"Blue")

            # BDLib.createGuideline(perspectFrame, res[0], res[1], outputDrawing)
            if res[0] is not None :
                # print(res)
                detectedBall = res[1]
                detectedBallPos = res[2]
                detectedBallTablePos = res[3]

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
                #print('K = ', k)
                if updatedBall[k] == '' :
                    updatedBall.pop(k)
                    updatedBallPos.pop(k)
                    updatedBallTablePos.pop(k)
                else :
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

            print('DetectedBall = ', detectedBall)
            # print('DetectedBallPos = ', detectedBallPos)
            # print('DetectedBallTablePos = ', detectedBallTablePos)
            print('UpdatedBall = ', updatedBall)
            print('UpdatedBallPos = ', updatedBallPos)
            print('UpdatedBallTablePos = ', updatedBallTablePos)
            print('BallProbs = ', ballProbs)
            if (timeTrialState == 'Playing') and (timeTrialMainBall in updatedBall) :
                if math.sqrt(pow((abs(int(updatedBallTablePos[updatedBall.index(timeTrialMainBall)][0]) - int(prevWhiteX))), 2) + pow(abs(int(updatedBallTablePos[updatedBall.index(timeTrialMainBall)][1]) - int(prevWhiteY)), 2)) >= 200 :
                    print("White Stop")
                    if ballCheckingStartTime == 0 :
                        ballCheckingStartTime = time.time()

                    if time.time() - ballCheckingStartTime > 2:
                        if 'Red' not in updatedBall :
                            timeTrialScore += 50
                        else:
                            timeTrialScore -= 10
                        prevWhiteX = updatedBallTablePos[updatedBall.index(timeTrialMainBall)][0]
                        prevWhiteY = updatedBallTablePos[updatedBall.index(timeTrialMainBall)][1] 
                        objectX = random.randint(20, 1900)
                        objectY = random.randint(20, 780)
                        ballCheckingStartTime = 0
                        timeTrialState = 'Positioning'
        else :
            print("Moving")

        if timeTrialState == 'Prepare':
            print('Prepare')
            cv2.circle(mac, (970, 440), 35, (255, 255, 255), 5)

            explanTextSize, explanTextBaseline = cv2.getTextSize(
                'Place cue ball inside the circle to start.', cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)

            cv2.putText(mac, 'Place cue ball inside the circle to start.', (970 - int((explanTextSize[0])/2), 550), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), 3, cv2.LINE_AA)
            
            print(timeTrialMainBall)
            if timeTrialMainBall in updatedBall :
                print('White : ', updatedBallTablePos[updatedBall.index(timeTrialMainBall)])
                if (updatedBallTablePos[updatedBall.index(timeTrialMainBall)][0] >= 935 and updatedBallTablePos[updatedBall.index(timeTrialMainBall)][0] <= 1005 and
                    updatedBallTablePos[updatedBall.index(timeTrialMainBall)][1] >= 405 and updatedBallTablePos[updatedBall.index(timeTrialMainBall)][1] <= 475) :
                    prevWhiteX = updatedBallTablePos[updatedBall.index(timeTrialMainBall)][0]
                    prevWhiteY = updatedBallTablePos[updatedBall.index(timeTrialMainBall)][1]
                    objectX = random.randint(20, 1900)
                    objectY = random.randint(20, 780)
                    timeTrialStartTime = time.time()
                    timeTrialState = 'Positioning'

            if cv2.waitKey(25) & 0xFF == ord('g'):
                prevWhiteX = updatedBallTablePos[updatedBall.index(timeTrialMainBall)][0]
                prevWhiteY = updatedBallTablePos[updatedBall.index(timeTrialMainBall)][1]
                objectX = random.randint(20, 1900)
                objectY = random.randint(20, 780)
                timeTrialState = 'Positioning'
        elif timeTrialState == 'Positioning' :
            print('Positioning')
            print('objectX', objectX)
            print('objectY', objectY)
            timeTextSize, timeTextBaseline = cv2.getTextSize(f'Time : {timeTrialMaxTime - int(time.time() - timeTrialStartTime)}', cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)
            scoreTextSize, scoreTextBaseline = cv2.getTextSize(f'Score : {timeTrialScore}', cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)

            cv2.putText(mac, f'Time : {timeTrialMaxTime - int(time.time() - timeTrialStartTime)}', (970 - int((timeTextSize[0])/2), 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    3.0, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(mac, f'Score : {timeTrialScore}', (970 - int((scoreTextSize[0])/2), 200), cv2.FONT_HERSHEY_SIMPLEX, 
                    3.0, (255, 255, 255), 5, cv2.LINE_AA)
                
            if (timeTrialMaxTime - int(time.time() - timeTrialStartTime)) == 0 :
                    timeTrialState = 'TimeOver'
            else :
                cv2.circle(mac, (objectX, objectY) , 35, (255, 255, 255), 5)

                if 'Red' in updatedBall :
                    print('Red : ', updatedBallTablePos[updatedBall.index(timeTrialObjBall)])
                    if (updatedBallTablePos[updatedBall.index(timeTrialObjBall)][0] >= objectX-35 and updatedBallTablePos[updatedBall.index(timeTrialObjBall)][0] <= objectX+35 and
                        updatedBallTablePos[updatedBall.index(timeTrialObjBall)][1] >= objectY-35 and updatedBallTablePos[updatedBall.index(timeTrialObjBall)][1] <= objectY+35) :
                        timeTrialState = 'Playing'
                    else :
                        timeTrialState = 'Positioning'
        elif timeTrialState == 'Playing' :
            print('Playing')
            timeTextSize, timeTextBaseline = cv2.getTextSize(f'Time : {timeTrialMaxTime - int(time.time() - timeTrialStartTime)}', cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)
            scoreTextSize, scoreTextBaseline = cv2.getTextSize(f'Score : {timeTrialScore}', cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)

            cv2.putText(mac, f'Time : {timeTrialMaxTime - int(time.time() - timeTrialStartTime)}', (970 - int((timeTextSize[0])/2), 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    3.0, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(mac, f'Score : {timeTrialScore}', (970 - int((scoreTextSize[0])/2), 200), cv2.FONT_HERSHEY_SIMPLEX, 
                    3.0, (255, 255, 255), 5, cv2.LINE_AA)
                
            if (timeTrialMaxTime - int(time.time() - timeTrialStartTime)) == 0 :
                    timeTrialState = 'TimeOver'
            else :
                cv2.circle(mac, (objectX, objectY) , 35, (0, 0, 255), 5)

        elif timeTrialState == 'TimeOver' :
            mac = cv2.imread('./pics/Stage/timeTrialResult.png')
            mac = cv2.resize(mac, (1920, 880))

            # Open a file for reading
            with open("HighScore.txt", "r") as f:
                # Read the contents of the file
                contents = f.read()

            # Print the contents of the file
            print(contents)

            timeTrialHighScore = contents

            if timeTrialScore > int(timeTrialHighScore) :
                timeTrialHighScore = timeTrialScore

                # Open a file for writing
                with open("HighScore.txt", "w") as f:
                    # Write some text to the file
                    f.write(str(timeTrialHighScore))
                
            highScoreTextSize, highScoreTextBaseline = cv2.getTextSize(f'{timeTrialHighScore}', cv2.FONT_HERSHEY_SIMPLEX, 4.0, 15)
            scoreSize, scoreBaseline = cv2.getTextSize(f'{timeTrialScore}', cv2.FONT_HERSHEY_SIMPLEX, 3.5, 15)

            cv2.putText(mac, f'{timeTrialHighScore}', (970 - int((highScoreTextSize[0])/2), 400 - int((highScoreTextSize[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                    4.0, (0, 0, 0), 15, cv2.LINE_AA)
            cv2.putText(mac, f'{timeTrialScore}', (970 - int((scoreSize[0])/2), 750 - int((scoreSize[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                    3.5, (0, 0, 0), 15, cv2.LINE_AA)

    if mainStage == 'multiplayerMode':
        succuess, frame = vidcap.read()
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
            cv2.circle(frame, i , 30, (255, 255, 255), -1)
        transformed_frame = cv2.warpPerspective(frame, M, (width, height))
        projection_frame = transformed_frame.copy()
        table_frame = transformed_frame[200:1080,0:1920]
        
        black = BDLib1.createTable()
        ret,thresh1 = cv2.threshold(black,127,255,cv2.THRESH_BINARY)
        
        # White close light 10 AM
        lower_green = np.array([50,0,0])
        upper_green = np.array([95,255,255])

        hsv = cv2.cvtColor(table_frame, cv2.COLOR_BGR2HSV)
        blurFrame = cv2.GaussianBlur(hsv, (7, 7), 0)
        mask = cv2.inRange(blurFrame, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # dilate->erode
        mask = cv2.dilate(mask_closing,kernel,iterations = 1)
        inv_mask = cv2.bitwise_not(mask)
        table_mask = inv_mask | thresh1
        cv2.imshow("0",table_mask)
        output = cv2.bitwise_and(table_frame,table_frame, mask= inv_mask)

        circles = cv2.HoughCircles(inv_mask, cv2.HOUGH_GRADIENT, 1, 30, param1=100, param2=10, minRadius=21, maxRadius=35)
        circleZones = []
        circleZonesColor = []

        res = BDLib.getCircles(handDetectFrame,"White")
        # BDLib.createGuideline(perspectFrame, res[0], res[1], outputDrawing)
        if res[0] is not None :
            # print(res)
            detectedBall = res[1]
            detectedBallPos = res[2]
            detectedBallTablePos = res[3]

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
            #print('K = ', k)
            if updatedBall[k] == '' :
                updatedBall.pop(k)
                updatedBallPos.pop(k)
                updatedBallTablePos.pop(k)
            else :
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
        # print('DetectedBall = ', detectedBall)
        # # print('DetectedBallPos = ', detectedBallPos)
        # # print('DetectedBallTablePos = ', detectedBallTablePos)
        #print('UpdatedBall = ', updatedBall)
        # print('UpdatedBallPos = ', updatedBallPos)
        print('UpdatedBallTablePos = ', updatedBallTablePos)
        # # print('BallProbs = ', ballProbs)
        # print('PrevBallPos = ' ,prevBallPos)
                # Threshold for detecting changes

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
                        cv2.putText(table_frame, f'Number : {i}', (circles[i][0], circles[i][1]-80), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(table_frame, f'Color : {similarColor}', (circles[i][0], circles[i][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(table_frame, f'X : {circles[i][0]}', (circles[i][0], circles[i][1]-30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(table_frame, f'Y : {circles[i][1]}', (circles[i][0], circles[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)

                        if similarColor == "Black":
                            realtime_black = [circles[i][0], circles[i][1],circles[i][2]]
                            if len(list_black) < 1:
                                list_black.append([circles[i][0], circles[i][1]])
                            # Define the points as numpy arrays
                            points_black = np.array(list_black)
                            # Calculate the mean of the points
                            avg_point_black  = np.mean(points_black , axis=0)
                            ax,ay = avg_point_black 
                            if abs(circles[i][0] - ax) < 20 and abs(circles[i][1] - ay) < 20 :
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
                            if abs(circles[i][0] - ax) < 20 and abs(circles[i][1] - ay) < 20 :
                                if len(list_white) < 3:
                                    list_white.append([circles[i][0], circles[i][1]])
                                else :
                                    avg_white = (int(ax)+10,int(ay))
                            else:
                                list_white.pop(0)

                    if "Black" not in detectedBall and  check_black is None:
                        check_black = time.time()
                    elif "Black" not in detectedBall and  check_black is not None:
                        if int(time.time()) - int(check_black) > 3 :
                                print("End Round")
                                print("Acurency Player1 : " + str(round((acurency_p1/count_shot_p1)*100)) + "  %")
                                print("Acurency Player2 : " + str(round((acurency_p2/count_shot_p2)*100)) + "  %")
                                check_black = None
                    else : 
                        check_black = None

                # for p in pocket_point:
                #     if len(realtime_black) != 0:
                #         if circleOverlap(realtime_black[0], realtime_black[1], realtime_black[2], p[0], p[1], 100) :
                #             print("End Round")
                #             print("Acurency Player1 : " + str(round((acurency_p1/count_shot_p1)*100)) + "  %")
                #             print("Acurency Player2 : " + str(round((acurency_p2/count_shot_p2)*100)) + "  %")
        if len(prevBallPos) == 0:
            prevBallPos = updatedBallTablePos

        threshold = 0.05
        # Comput the distance between each ball's previous position and new position
        distances = []
        
        for prev in prevBallPos:
            # Find the closest new position to the previous position
            closest_new = min(updatedBallTablePos, key=lambda new: math.sqrt((int(prev[0])-int(new[0]))**2 + (int(prev[1])-int(new[1]))**2))
            # Compute the distance between the closest new position and the previous position
            distance = math.sqrt( (int(prev[0])-int(closest_new[0]) )**2 + ( int(prev[1])-int(closest_new[1]) )**2)
            distances.append(distance)
            print(closest_new)

            
        # Check if any distance is above the threshold
        if any(distance > threshold for distance in distances):
            print("There has been a change in the ball positions.")
            break
        else:
            print("The ball positions have not changed much.")

        whitePos = -1
        if 'White' in detectedBall:
            whitePos = detectedBall.index('White')
        #print(realtime_black)
        if whitePos != -1 :
            # if isP1:
            #     if prevBallPos != updatedBall:
            #         acurency_p1 += 1
            # else :
            #     if prevBallPos != updatedBall:
            #         acurency_p2 += 1
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
                #prevBallPos = updatedBallTablePos
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

                if len(list_start) < 1:
                    list_start.append([start_x,start_y])
                    list_end.append([end_x,end_y])

                cue_1 = np.array(list_start)
                cue_2 = np.array(list_end)
                # Calculate the mean of the points
                avg_cue1 = np.mean(cue_1, axis=0)
                avg_cue2 = np.mean(cue_2, axis=0)
                ax_start,ay_start = avg_cue1
                ax_end,ay_end = avg_cue2

                if abs(start_x - ax_start) < 100 and abs(start_y - ay_start) < 100 and abs(end_x - ax_end) < 100 and abs(end_y - ay_end) < 100 :
                    if len(list_start) < 5 :
                        list_start.append([start_x,start_y])
                        list_end.append([end_x,end_y])
                    else:
                        cv2.line(black, (round(ax_start), round(ay_start)), (round(ax_end), round(ay_end)), (255, 255, 255), 5)
                        isDraw = True
                else:
                    list_start.clear()
                    list_end.clear()
                    isDraw = False

                output = cv2.bitwise_and(whiteball_zone, whiteball_zone, mask=mask)
                edges = cv2.Canny(output, 180, 255)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 45,minLineLength=10, maxLineGap=100)
                if lines is not None and isDraw == True:
                    start_x,start_y = round(ax_start), round(ay_start)
                    end_x,end_y = round(ax_end), round(ay_end)
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
        else :
            list_start.clear()
            list_end.clear()
        black_bg = np.zeros((1080, 1920, 3), np.uint8)
        black_bg[200:1080,0:1920] = black
        # cv2.namedWindow('Frame',cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow("Frame", black_bg)
        #cv2.imshow("Frame1", table_frame)
        #cv2.imshow("Frame2", black)
        #cv2.imshow("Frame3", whiteball_zone)
        detectedBall.clear()
        detectedBallPos.clear()
        detectedBallTablePos.clear()
        tansformed_frame = black_bg

    if mainStage == 'freedomMode':

        succuess, frame = vidcap.read()
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
            cv2.circle(frame, i , 30, (255, 255, 255), -1)
        transformed_frame = cv2.warpPerspective(frame, M, (width, height))
        projection_frame = transformed_frame.copy()
        table_frame = transformed_frame[200:1080,0:1920]
        
        black = BDLib1.createTable()
        ret,thresh1 = cv2.threshold(black,127,255,cv2.THRESH_BINARY)
        
        # White close light 10 AM
        lower_green = np.array([50,20,40])
        upper_green = np.array([95,255,255])

        hsv = cv2.cvtColor(table_frame, cv2.COLOR_BGR2HSV)
        blurFrame = cv2.GaussianBlur(hsv, (7, 7), 0)
        mask = cv2.inRange(blurFrame, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # dilate->erode
        mask = cv2.dilate(mask_closing,kernel,iterations = 1)
        inv_mask = cv2.bitwise_not(mask)
        table_mask = inv_mask | thresh1
        cv2.imshow("0",table_mask)
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
                        cv2.putText(table_frame, f'Number : {i}', (circles[i][0], circles[i][1]-80), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(table_frame, f'Color : {similarColor}', (circles[i][0], circles[i][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(table_frame, f'X : {circles[i][0]}', (circles[i][0], circles[i][1]-30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(table_frame, f'Y : {circles[i][1]}', (circles[i][0], circles[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)

                        if similarColor == "Black":
                            realtime_black = [circles[i][0], circles[i][1],circles[i][2]]
                            if len(list_black) < 1:
                                list_black.append([circles[i][0], circles[i][1]])
                            # Define the points as numpy arrays
                            points_black = np.array(list_black)
                            # Calculate the mean of the points
                            avg_point_black  = np.mean(points_black , axis=0)
                            ax,ay = avg_point_black 
                            if abs(circles[i][0] - ax) < 20 and abs(circles[i][1] - ay) < 20 :
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
                            if abs(circles[i][0] - ax) < 20 and abs(circles[i][1] - ay) < 20 :
                                if len(list_white) < 3:
                                    list_white.append([circles[i][0], circles[i][1]])
                                else :
                                    avg_white = (int(ax)+10,int(ay))
                            else:
                                list_white.pop(0)

                    if "Black" not in detectedBall and  check_black is None:
                        check_black = time.time()
                    elif "Black" not in detectedBall and  check_black is not None:
                        if int(time.time()) - int(check_black) > 3 :
                                print("End Round")
                                print("Acurency Player1 : " + str(round((acurency_p1/count_shot_p1)*100)) + "  %")
                                print("Acurency Player2 : " + str(round((acurency_p2/count_shot_p2)*100)) + "  %")
                                check_black = None
                    else : 
                        check_black = None

                # for p in pocket_point:
                #     if len(realtime_black) != 0:
                #         if circleOverlap(realtime_black[0], realtime_black[1], realtime_black[2], p[0], p[1], 100) :
                #             print("End Round")
                #             print("Acurency Player1 : " + str(round((acurency_p1/count_shot_p1)*100)) + "  %")
                #             print("Acurency Player2 : " + str(round((acurency_p2/count_shot_p2)*100)) + "  %")

            

        whitePos = -1
        if 'White' in detectedBall:
            whitePos = detectedBall.index('White')
        #print(realtime_black)
        if whitePos != -1 :
            # if isP1:
            #     if prevBallPos != updatedBall:
            #         acurency_p1 += 1
            # else :
            #     if prevBallPos != updatedBall:
            #         acurency_p2 += 1
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
                #prevBallPos = updatedBallTablePos
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

                if len(list_start) < 1:
                    list_start.append([start_x,start_y])
                    list_end.append([end_x,end_y])

                cue_1 = np.array(list_start)
                cue_2 = np.array(list_end)
                # Calculate the mean of the points
                avg_cue1 = np.mean(cue_1, axis=0)
                avg_cue2 = np.mean(cue_2, axis=0)
                ax_start,ay_start = avg_cue1
                ax_end,ay_end = avg_cue2

                if abs(start_x - ax_start) < 100 and abs(start_y - ay_start) < 100 and abs(end_x - ax_end) < 100 and abs(end_y - ay_end) < 100 :
                    if len(list_start) < 5 :
                        list_start.append([start_x,start_y])
                        list_end.append([end_x,end_y])
                    else:
                        cv2.line(black, (round(ax_start), round(ay_start)), (round(ax_end), round(ay_end)), (255, 255, 255), 5)
                        isDraw = True
                else:
                    list_start.clear()
                    list_end.clear()
                    isDraw = False

                output = cv2.bitwise_and(whiteball_zone, whiteball_zone, mask=mask)
                edges = cv2.Canny(output, 180, 255)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 45,minLineLength=10, maxLineGap=100)
                if lines is not None and isDraw == True:
                    start_x,start_y = round(ax_start), round(ay_start)
                    end_x,end_y = round(ax_end), round(ay_end)
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
        else :
            list_start.clear()
            list_end.clear()
        black_bg = np.zeros((1080, 1920, 3), np.uint8)
        black_bg[200:1080,0:1920] = black
        # cv2.namedWindow('Frame',cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow("Frame", black_bg)
        #cv2.imshow("Frame1", table_frame)
        #cv2.imshow("Frame2", black)
        #cv2.imshow("Frame3", whiteball_zone)
        detectedBall.clear()
        detectedBallPos.clear()
        detectedBallTablePos.clear()
        tansformed_frame = black_bg
        # print('mainStage : ', mainStage)
    
    #bgFrame = motionBlurFrame
    cv2.namedWindow('Test_Perspectice',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Test_Perspectice', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Test_Perspectice", tansformed_frame)
    # cv2.imshow('Hand Zone', handDetectFrame)
    #cv2.imshow("Test", frame)

    
    if cv2.waitKey(1) == 27:
        break
