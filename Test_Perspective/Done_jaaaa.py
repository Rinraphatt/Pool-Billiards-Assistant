import numpy as np
import cv2
import time

vidcap = cv2.VideoCapture(0)
width = 1920
height = 1080
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

# Define the lower and upper bounds of the skin color in the HSV color space
lower_skin = np.array([80, 80, 80], dtype=np.uint8)
upper_skin = np.array([100, 255, 255], dtype=np.uint8)

# Define the rectangular zone
x1NextBtn, y1NextBtn = 1750, 710  # top-left corner
x2NextBtn, y2NextBtn = 1850, 810  # bottom-right corner

x1PrevBtn, y1PrevBtn = 1600, 710  # top-left corner
x2PrevBtn, y2PrevBtn = 1700, 810  # bottom-right corner

x1BackBtn, y1BackBtn = 1750, 70  # top-left corner
x2BackBtn, y2BackBtn = 1850, 170  # bottom-right corner

x1TrainingBtn, y1TrainingBtn = 611, 105  # top-left corner
x2TrainingBtn, y2TrainingBtn = 1310, 403  # bottom-right corner

x1PlayBtn, y1PlayBtn = 611, 477  # top-left corner
x2PlayBtn, y2PlayBtn = 1310, 775  # bottom-right corner

x1BasicBtn, y1BasicBtn = 234, 394  # top-left corner
x2BasicBtn, y2BasicBtn = 681, 763  # bottom-right corner

x1AmatureBtn, y1AmatureBtn = 737, 394  # top-left corner
x2AmatureBtn, y2AmatureBtn = 1184, 763  # bottom-right corner

x1ProBtn, y1ProBtn = 1239, 394  # top-left corner
x2ProBtn, y2ProBtn = 1686, 763  # bottom-right corner


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
# mac = cv2.imread('./pics/Stage/modeSelect')
mac = cv2.imread('./pics/Stage/modeSelect.png')
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
        # cv2.rectangle(handDetectFrame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(tansformed_frame, (x, y+200), (x+w, y+h+200), (0, 255, 0), 2)

        # Create Button Rectangle
        cursorRect = x,y,w,h
        nextBtnRect = x1NextBtn,y1NextBtn,x2NextBtn-x1NextBtn,y2NextBtn-y1NextBtn
        prevBtnRect = x1PrevBtn,y1PrevBtn,x2PrevBtn-x1PrevBtn,y2PrevBtn-y1PrevBtn
        backBtnRect = x1BackBtn,y1BackBtn,x2BackBtn-x1BackBtn,y2BackBtn-y1BackBtn
        trainingBtnRect = x1TrainingBtn,y1TrainingBtn,x2TrainingBtn-x1TrainingBtn,y2TrainingBtn-y1TrainingBtn
        basicBtnRect = x1BasicBtn,y1BasicBtn,x2BasicBtn-x1BasicBtn,y2BasicBtn-y1BasicBtn
        amatureBtnRect = x1AmatureBtn,y1AmatureBtn,x2AmatureBtn-x1AmatureBtn,y2AmatureBtn-y1AmatureBtn
        proBtnRect = x1ProBtn,y1ProBtn,x2ProBtn-x1ProBtn,y2ProBtn-y1ProBtn
        leftStageBtnRect = basicBtnRect
        middleStageBtnRect = amatureBtnRect
        rightStageBtnRect = proBtnRect

        # Draw the rectangular zone on the frame
        # cv2.rectangle(handDetectFrame, (x1NextBtn, y1NextBtn), (x2NextBtn, y2NextBtn), (0, 255, 0), 3)
        # cv2.rectangle(handDetectFrame, (x1PrevBtn, y1PrevBtn), (x2PrevBtn, y2PrevBtn), (255, 0, 0), 3)
        if mainStage == 'modeSelect' :
            cv2.rectangle(tansformed_frame, (x1TrainingBtn, y1TrainingBtn+200), (x2TrainingBtn, y2TrainingBtn+200), (255, 0, 0), 3)
        elif mainStage == 'difficultSelect' or mainStage == 'basicStageSelect' or mainStage == 'amatureStageSelect' or mainStage == 'proStageSelect':
            cv2.rectangle(tansformed_frame, (x1BasicBtn, y1BasicBtn+200), (x2BasicBtn, y2BasicBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1AmatureBtn, y1AmatureBtn+200), (x2AmatureBtn, y2AmatureBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1ProBtn, y1ProBtn+200), (x2ProBtn, y2ProBtn+200), (255, 0, 0), 3)
            cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 0, 255), 3)
        elif mainStage == 'trainingStage' :
            cv2.rectangle(tansformed_frame, (x1NextBtn, y1NextBtn+200), (x2NextBtn, y2NextBtn+200), (0, 255, 0), 3)
            cv2.rectangle(tansformed_frame, (x1PrevBtn, y1PrevBtn+200), (x2PrevBtn, y2PrevBtn+200), (255, 0, 0), 3)
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
                    mac = cv2.imread('../pics/Stage/proStageSelect.png')
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
                    mac = cv2.imread('../pics/Stage/' + currentStagePics[currentStageState])
                    mac = cv2.resize(mac, (1920, 880))
                    start_time = time.time()
            elif are_rectangles_overlapping(cursorRect,backBtnRect) == True :
                if start_time is None:
                    start_time = time.time()

                elif time.time() - start_time >= debounceTime:
                    cv2.rectangle(tansformed_frame, (x1BackBtn, y1BackBtn+200), (x2BackBtn, y2BackBtn+200), (0, 255, 255), 3)
                    prevMainStage = 'modeSelect'
                    mainStage = 'difficultSelect'
                    mac = cv2.imread('../pics/Stage/difficultSelect.png') 
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
                    mac = cv2.imread('../pics/Stage/' + currentStagePics[currentStageState])
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
                    mac = cv2.imread('../pics/Stage/' + currentStagePics[currentStageState])
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
                    mac = cv2.imread('../pics/Stage/' + currentStagePics[currentStageState])
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
  
    cv2.namedWindow('Test_Perspectice',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Test_Perspectice', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Test_Perspectice", tansformed_frame)
    # cv2.imshow('Hand Zone', handDetectFrame)
    #cv2.imshow("Test", frame)

    
    if cv2.waitKey(1) == 27:
        break
