import cv2 as cv
import numpy as np

outputDrawing = np.zeros((784,1568,3), np.uint8)

while True:
    # success, img = cam.read()
    # success2, img2 = cam2.read()
    # frame = img
    # showFrame = img2

    # ret, frame = cam.read()
    frame = cv.imread('../pics/pool_table_ball.jpg')
    frame = cv.resize(frame, (1920, 1080))

    showFrame = cv.imread('../pics/pool_table_ball.jpg')
    showFrame = cv.resize(frame, (1920, 1080))

    cropped_image = frame[148:932, 174:1742]

    hsv = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)

    lower_green = np.array([30,30,20])
    upper_green = np.array([70,255,255])

    mask = cv.inRange(hsv, lower_green, upper_green)

    res = cv.bitwise_and(cropped_image, cropped_image, mask=mask)

    circleZones = []
    circleZonesColor = []
    circleGrayScaleValues = []

    blurFrame = cv.GaussianBlur(mask, (7,7), 0)

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 100,
                                param1=100, param2=30, minRadius=5, maxRadius=30)
    
    print(circles)
     
    if circles is not None:
        circles = np.uint16(np.around(circles))

        circleCounter = 0

        # print(circles[0])

        for i in circles[0, :]:
            # cv.circle(frame, (i[0], i[1]), 1, (0,200,200), 2)
            # cv.circle(frame, (i[0], i[1]), i[2], (255,0,255), 2)
            circleZone = blurFrame[148+int(i[1].item())-22:148+int(i[1].item())+22, 174+int(i[0].item())-22:174+int(i[0].item())+22]
            circleZones.append(circleZone)

            circleZoneColor = frame[148+int(i[1].item())-22:148+int(i[1].item())+22, 174+int(i[0].item())-22:174+int(i[0].item())+22]
            circleZonesColor.append(circleZoneColor)

            cv.circle(showFrame, (i[0]+174, i[1]+148), i[2], (255,0,255), 2)
            # cv.circle(outputDrawing, (i[0]-300, i[1]-200), i[2], (255,0,255), 2)

            circleCounterInner = 0
            for j in circles[0, :]:
                cv.line(outputDrawing,(int(round(circles[0][circleCounterInner][0]-300)),int(round(circles[0][circleCounterInner][1]-200))),(int(round(circles[0][circleCounter][0]-300)),int(round(circles[0][circleCounter][1]-200))),(0,255,0),2)
                circleCounterInner += 1

            # cropped_image = blurFrame[i[0]-200: i[0]+200, i[1]-200: i[1]+200]
            circleCounter += 1

    # cv.imshow("CroppedShowFrame", cropped_image)
    cv.imshow('mask', mask)
    cv.imshow('blurFrame', blurFrame)
    cv.imshow('res', res)
    cv.imshow('showFrame', showFrame)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cv.destroyAllWindows()