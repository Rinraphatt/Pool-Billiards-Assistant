import numpy as np
import cv2 as cv

vidcap = cv.VideoCapture("Test_Perspective/newVid.mp4")
succuess, img = vidcap.read()
width = 1280
heigth = 720
while succuess:
    succuess, img = vidcap.read()
    frame = img

    tl = (177 ,159)
    bl = (180 ,922)
    tr = (1741 ,155)
    br = (1742 ,925)
    cv.circle(frame, tl, 3, (0, 0, 255), -1)
    cv.circle(frame, bl, 3, (0, 0, 255), -1)
    cv.circle(frame, tr, 3, (0, 0, 255), -1)
    cv.circle(frame, br, 3, (0, 0, 255), -1)
    cv.line(frame, tl, bl, (0, 255, 0), 2)
    cv.line(frame, bl, br, (0, 255, 0), 2)
    cv.line(frame, br, tr, (0, 255, 0), 2)
    cv.line(frame, tl, tr, (0, 255, 0), 2)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, heigth], [width, 0], [width, heigth]])
    
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    tansformed_frame = cv.warpPerspective(frame, matrix, (width, heigth))

    grayFrame = cv.cvtColor(tansformed_frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (7,7), 0)
    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 100,
                                param1=100, param2=30, minRadius=5, maxRadius=30)
    circleZones = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))

        circleCounter = 0

        # print(circles[0])
        
        for i in circles[0, :]:
            # cv.circle(frame, (i[0], i[1]), 1, (0,200,200), 2)
            # cv.circle(frame, (i[0], i[1]), i[2], (255,0,255), 2)

            cv.circle(tansformed_frame, (i[0], i[1]), i[2], (255,0,255), 2)
            # cv.circle(frame, (i[0], i[1]), 30, (255,0,255), 2)
            #cv.circle(outputDrawing, (i[0]-300, i[1]-200), i[2], (255,0,255), 2)


            circleZone = blurFrame[148+int(i[1].item())-20:148+int(i[1].item())+20, 174+int(i[0].item())-20:174+int(i[0].item())+20]
            circleZones.append(circleZone)

            circleCounterInner = 0
            # for j in circles[0, :]:
            #     cv.line(outputDrawing,(int(round(circles[0][circleCounterInner][0]-300)),int(round(circles[0][circleCounterInner][1]-200))),(int(round(circles[0][circleCounter][0]-300)),int(round(circles[0][circleCounter][1]-200))),(0,255,0),2)
            #     circleCounterInner += 1

            # cropped_image = blurFrame[i[0]-200: i[0]+200, i[1]-200: i[1]+200]
            circleCounter += 1
    
    cv.imshow("Test_Perspectice", tansformed_frame)

    cv.imshow("Test", frame)
    
    if cv.waitKey(1) == 27:
        break
