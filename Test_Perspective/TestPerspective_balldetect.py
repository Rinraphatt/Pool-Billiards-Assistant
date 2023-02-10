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
            
    if circles is not None:
        whiteValue = -1000000
        whiteZone = []

        whitePos = 0

        for i in range(len(circleZones)):
            testFrame = circleZones[i]
            print(testFrame)
            if testFrame != [] :
                avg_color_per_row = np.average(testFrame, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                print("CircleZones" + str(i) + " : " + str(avg_color))
                if avg_color > whiteValue : 
                    whiteValue = avg_color
                    whiteZone = circleZones[i]
                    whitePos = i
                # cv.imshow("CircleZones" + str(i), circleZones[i])

        print(whiteValue)

        # cv.imshow("WhiteCircleZone", whiteZone)

        print(whitePos)
        # print(circles[0])
        print(circles[0][whitePos][0])
        print(circles[0][whitePos][1])

        cropped_whiteZone = frame[260+circles[0][whitePos][1]-100: 260+circles[0][whitePos][1]+100, -193+174+circles[0][whitePos][0]-100: -193+174+circles[0][whitePos][0]+100]
        edges = cv.Canny(cropped_whiteZone, 50, 200)
        # Detect points that form a line
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=5, maxLineGap=250)
        #print(lines)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(cropped_whiteZone, (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv.imshow("RealWhiteCircleZone", cropped_whiteZone)
        # cv.imshow("SolidCircleZone1", frame[148+circles[0][3][1]-100: 148+circles[0][3][1]+100, 174+circles[0][3][0]-100: 174+circles[0][3][0]+100])
        # cv.imshow("SolidCircleZone2", frame[148+circles[0][6][1]-100: 148+circles[0][6][1]+100, 174+circles[0][6][0]-100: 174+circles[0][6][0]+100])
        # cv.imshow("SolidCircleZone3", frame[148+circles[0][12][1]-100: 148+circles[0][12][1]+100, 174+circles[0][12][0]-100: 174+circles[0][12][0]+100])
        # cv.imshow("SolidCircleZone4", frame[148+circles[0][8][1]-100: 148+circles[0][8][1]+100, 174+circles[0][8][0]-100: 174+circles[0][8][0]+100])
        # cv.imshow("SolidCircleZone5", frame[148+circles[0][10][1]-100: 148+circles[0][10][1]+100, 174+circles[0][10][0]-100: 174+circles[0][10][0]+100])
        # cv.imshow("SolidCircleZone6", frame[148+circles[0][14][1]-100: 148+circles[0][14][1]+100, 174+circles[0][14][0]-100: 174+circles[0][14][0]+100])
        # cv.imshow("SolidCircleZone7", frame[148+circles[0][9][1]-100: 148+circles[0][9][1]+100, 174+circles[0][9][0]-100: 174+circles[0][9][0]+100])
        

    circleZones = []
    cv.imshow("Test_Perspectice", tansformed_frame)
    frame = cv.resize(frame, (192*3, 108*3))
    cv.imshow("Test", frame)
    
    if cv.waitKey(1) == 27:
        break
