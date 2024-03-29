
from dis import dis
from tkinter import Frame
from tkinter.colorchooser import Chooser
from turtle import circle
import cv2 as cv
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

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

cameraHeight=1080 
cameraWidth=1920

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture('./videos/Test.mp4')
# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, cameraHeight)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, cameraWidth)
prevCircle = None
cropSize = (100, 100)

# cv.namedWindow("Python Webcam Screenshot App")
def createBar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

# def compareWithRGBColors(dominantColor):
#     minDiffValue = 1000
#     minDiffPos = -1
#     for j in range(len(rgbColors)):
#         pctDiffRed = abs(dominantColor[0]-rgbColors[j][0]) / 255
#         pctDiffGreen = abs(dominantColor[1]-rgbColors[j][1]) / 255
#         pctDiffBlue = abs(dominantColor[2]-rgbColors[j][2]) / 255

#         pctDiffRGB = (pctDiffRed + pctDiffGreen + pctDiffBlue) / 3 * 100

#         if pctDiffRGB < minDiffValue:
#             minDiffPos = j
#             minDiffValue = pctDiffRGB
#     return minDiffPos

def compareWithRGBColors(dominantColor):
    minDiffValue = 1000
    minDiffPos = -1
    for i in range(len(rgbColors)):
        r1 = dominantColor[2] / 255
        g1 = dominantColor[1] / 255
        b1 = dominantColor[0] / 255

        r2 = rgbColors[i][2] / 255
        g2 = rgbColors[i][1] / 255
        b2 = rgbColors[i][0] / 255

        # Red Color
        color1_rgb = sRGBColor(r1, g1, b1)

        # Blue Color
        color2_rgb = sRGBColor(r2, g2, b2)

        # Convert from RGB to Lab Color Space
        color1_lab = convert_color(color1_rgb, LabColor)

        # Convert from RGB to Lab Color Space
        color2_lab = convert_color(color2_rgb, LabColor)

        # Find the color difference
        delta_e = delta_e_cie2000(color1_lab, color2_lab)

        if delta_e < minDiffValue:
            minDiffPos = i
            minDiffValue = delta_e
    return minDiffPos

outputDrawing = np.zeros((784,1568,3), np.uint8)

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

# rgbColors = [
#     [17.0, 163.0, 212.0],
#     [92.0, 53.0, 37.0],
#     [43.0, 76.0, 201.0],
#     [109.0, 72.0, 87.0],
#     [50.0, 136.0, 238.0],
#     [106.0, 140.0, 47.0],
#     [55.0, 77.0, 87.0],
#     [255.0, 255.0, 255.0],
#     [50.0, 50.0, 50.0]
# ]

rgbColors = [
    [17.0, 163.0, 212.0], #Yellow
    [92.0, 53.0, 37.0], #Blue
    [54.0, 83.0, 197.0], #Red
    [129.0, 72.0, 107.0], #Purple
    [50.0, 136.0, 238.0], #Orange
    [106.0, 140.0, 47.0], #Green
    [55.0, 77.0, 132.0], #Crimson
    [55.0, 55.0, 55.0], #Black
    [200.0, 200.0, 200.0] #White
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

# loadSetting()

while True:
    # success, img = cam.read()
    # success2, img2 = cam2.read()
    # frame = img
    # showFrame = img2

    # ret, frame = cam.read()
    frame = cv.imread('../pics/pool_table_ball.jpg')
    frame = cv.resize(frame, (1920, 1080))
    # print('frame', frame)

    showFrame = cv.imread('../pics/pool_table_ball.jpg')
    showFrame = cv.resize(frame, (1920, 1080))

    # if not ret: break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (7,7), 0)

    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # print('hsvFrame', hsvFrame)

    # circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 100,
    #                             param1=100, param2=30, minRadius=10, maxRadius=25)

    cropped_image = frame[148:932, 174:1742]
    cropped_Blur_image = blurFrame[148:932, 174:1742]
    cropped_Show_image = showFrame[148:932, 174:1742]

    circles = cv.HoughCircles(cropped_Blur_image, cv.HOUGH_GRADIENT, 1.4, 100,
                                param1=100, param2=30, minRadius=5, maxRadius=30)

    circleZones = []
    circleZonesColor = []
    circleGrayScaleValues = []
     
    if circles is not None:
        circles = np.uint16(np.around(circles))

        circleCounter = 0

        # print(circles[0])

        for i in circles[0, :]:
            # cv.circle(frame, (i[0], i[1]), 1, (0,200,200), 2)
            # cv.circle(frame, (i[0], i[1]), i[2], (255,0,255), 2)
            circleZone = blurFrame[148+int(i[1].item())-20:148+int(i[1].item())+20, 174+int(i[0].item())-20:174+int(i[0].item())+20]
            circleZones.append(circleZone)

            circleZoneColor = frame[148+int(i[1].item())-20:148+int(i[1].item())+20, 174+int(i[0].item())-20:174+int(i[0].item())+20]
            circleZonesColor.append(circleZoneColor)

            cv.circle(showFrame, (i[0]+174, i[1]+148), i[2], (255,0,255), 2)
            # cv.circle(outputDrawing, (i[0]-300, i[1]-200), i[2], (255,0,255), 2)

            circleCounterInner = 0
            for j in circles[0, :]:
                cv.line(outputDrawing,(int(round(circles[0][circleCounterInner][0]-300)),int(round(circles[0][circleCounterInner][1]-200))),(int(round(circles[0][circleCounter][0]-300)),int(round(circles[0][circleCounter][1]-200))),(0,255,0),2)
                circleCounterInner += 1

            # cropped_image = blurFrame[i[0]-200: i[0]+200, i[1]-200: i[1]+200]
            circleCounter += 1

    # cv.imshow("Output", outputDrawing)

    if circles is not None:
        whiteZone = []

        whitePos = -1

        # Find GrayScaleValue Of Every Balls and Find White Ball
        for i in range(len(circleZones)):
            testFrame = circleZones[i]
            avg_color_per_row = np.average(testFrame, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            circleGrayScaleValues.append(round(avg_color))

            cv.putText(showFrame, f'{i} : {round(avg_color)}', (circles[0][i][0]+120, circles[0][i][1]+200), cv.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 255), 2, cv.LINE_AA)
            # if avg_color > whiteValue : 
            #     whiteValue = avg_color
            #     whiteZone = circleZones[i]
            #     whitePos = i

        # Find Color and Type of Every Balls
        for i in range(len(circleZones)):
            height, width, _ = np.shape(circleZonesColor[i])
            data = np.reshape(circleZonesColor[i], (height * width, 3))
            data = np.float32(data)

            number_clusters = 2
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv.KMEANS_RANDOM_CENTERS
            compactness, labels, centers = cv.kmeans(data, number_clusters, None, criteria, 10, flags)
            # print(centers)

            cluster_sizes = np.bincount(labels.flatten())

            palette = []
            sortedDominantColor = []

            for cluster_idx in np.argsort(-cluster_sizes):
                sortedDominantColor.append(centers[cluster_idx])
                palette.append(np.full((circleZonesColor[i].shape[0], circleZonesColor[i].shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
            palette = np.hstack(palette)

            sf = circleZonesColor[i].shape[1] / palette.shape[1]
            out = np.vstack([circleZonesColor[i], cv.resize(palette, (0, 0), fx=sf, fy=sf)])


            # cv.imshow("dominant_colors", out)

            print(f'{i}-1 : {sortedDominantColor[0]}')

            print(f'WhitePos : {whitePos}')

            # minDiffPos = compareWithRGBColors(sortedDominantColor[0])
            # print(f'MinDiffPos : {minDiffPos}')
            # if minDiffPos == 8 and i != whitePos:
            #     minDiffPos = compareWithRGBColors(sortedDominantColor[1])
            #     print(f'Change MinDiffPos to : {minDiffPos}')

            realDominantColor = ''
            minDiffPos = compareWithRGBColors(sortedDominantColor[0])
            realDominantColor = sortedDominantColor[0]
            print(f'MinDiffPos : {minDiffPos}')
            if minDiffPos == 8:
                if circleGrayScaleValues[i] >= grayScaleValues[8][1]:
                    whitePos = i
                    whiteZone = circleZones[i]
                else:
                    minDiffPos = compareWithRGBColors(sortedDominantColor[1])
                    realDominantColor = sortedDominantColor[1]
                print(f'Change MinDiffPos to : {minDiffPos}')

            similarColor = ''
            # print(f'Less Diff RGB : {minDiffPos}') 
            if minDiffPos == 0:
                similarColor = 'Yellow'
            elif minDiffPos == 1:
                similarColor = 'Blue'
            elif minDiffPos == 2:
                similarColor = 'Red'
            elif minDiffPos == 3:
                similarColor = 'Purple'
            elif minDiffPos == 4:
                similarColor = 'Orange'
            elif minDiffPos == 5:
                similarColor = 'Green'
            elif minDiffPos == 6:
                similarColor = 'Crimson'
            elif minDiffPos == 7:
                similarColor = 'Black'
            elif minDiffPos == 8:
                similarColor = 'White'

            print(f'Similar to {similarColor}')

            colorCounter = 0
            lower_blue = np.array([110,50,50])
            upper_blue = np.array([130,255,255])

            # print(circleZonesColor[i])
            hsvcircleZone = cv.cvtColor(circleZonesColor[i], cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsvcircleZone, lower_blue, upper_blue)
            whitePixs = np.sum(mask == 255)
            print('Number of blue pixels : ', whitePixs)
            
            cv.putText(showFrame, f'Color : {similarColor}', (circles[0][i][0]+120, circles[0][i][1]+230), cv.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 0, 255), 2, cv.LINE_AA)
            if (minDiffPos >= 0 and minDiffPos <= 6):
                if circleGrayScaleValues[i] <= grayScaleValues[minDiffPos][0]:
                    cv.putText(showFrame, f'Type : Solid', (circles[0][i][0]+120, circles[0][i][1]+250), cv.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 0, 255), 2, cv.LINE_AA)
                else:
                    cv.putText(showFrame, f'Type : Stripe', (circles[0][i][0]+120, circles[0][i][1]+250), cv.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 0, 255), 2, cv.LINE_AA)
            # cv.putText(showFrame, f'Color : {sortedDominantColor[0]}', (circles[0][i][0]+120, circles[0][i][1]+250), cv.FONT_HERSHEY_SIMPLEX, 
            #     0.7, (255, 0, 255), 2, cv.LINE_AA)
            # cv.putText(showFrame, f'Color : {sortedDominantColor[1]}', (circles[0][i][0]+120, circles[0][i][1]+270), cv.FONT_HERSHEY_SIMPLEX, 
            #     0.7, (255, 0, 255), 2, cv.LINE_AA)
            # cv.putText(showFrame, f'Color : {realDominantColor}', (circles[0][i][0]+120, circles[0][i][1]+270), cv.FONT_HERSHEY_SIMPLEX, 
            #     0.7, (255, 0, 255), 2, cv.LINE_AA)

        # print(whiteValue)

        # cv.imshow("WhiteCircleZone", whiteZone)

        print(whitePos)
        # print(circles[0])
        print(circles[0][whitePos][0])
        print(circles[0][whitePos][1])

        cropped_whiteZone = frame[148+circles[0][whitePos][1]-100: 148+circles[0][whitePos][1]+100, 174+circles[0][whitePos][0]-100: 174+circles[0][whitePos][0]+100]
        #cv.imshow("RealWhiteCircleZone", cropped_whiteZone)

        # Convert the frame to HSV color space
        hsv = cv.cvtColor(cropped_whiteZone, cv.COLOR_BGR2HSV)


        # Define a cue white color threshold
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([179, 60, 255])

        mask = cv.inRange(hsv, lower_white, upper_white)
        output = cv.bitwise_and(cropped_whiteZone,cropped_whiteZone, mask= mask)

        cv.imshow("test", output)



        edges = cv.Canny(output, 130, 255)
        # Detect points that form a line
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=10, maxLineGap=100)
        # Draw the detected line segments on the original frame
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(output, (x1, y1), (x2, y2), (0, 0, 255), 1)
                # Calculate the center of two line segments

            # Find the distance between the two parallel lines
            if len(lines) == 2:
                #print(lines)

                x1, y1, x2, y2 = lines[0][0]
                x3, y3, x4, y4 = lines[1][0]
                # cv.circle(cropped_whiteZone, (x1, y1), 2, (0, 255, 255), -1)
                # cv.circle(cropped_whiteZone, (x2, y2), 2, (0, 255, 255), -1)
                # cv.circle(cropped_whiteZone, (x3, y3), 2, (0, 255, 255), -1)
                # cv.circle(cropped_whiteZone, (x4, y4), 2, (0, 255, 255), -1)

                m1 = (round((x1+x3) /2) , round((y1+y3) /2)) 
                m3 = (round((x2+x4) /2) , round((y2+y4) /2))
                #print(m1)
                cv.circle(cropped_whiteZone, m1, 1, (255, 255, 0), -1)
                cv.circle(cropped_whiteZone, m3, 1, (255, 0, 0), -1) 

                x1, y1 = m1
                x2, y2 = m3
                # Calculate the slope and intercept of the line
                m = 0
                if (x2-x1) != 0 :
                    m = (y2 - y1) / (x2 - x1)
                else:
                    m = (y2 - y1) / 0.01
                b = y1 - m * x1
                print(b)

                # Define the image boundaries
                height, width = cropped_whiteZone.shape[:2]
                #print(height, width)
                y_top = 0
                y_bottom = height - 1
                x_left = 0
                x_right = width - 1

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

                # Draw the line segment between the two points
                # cv.line(cropped_whiteZone, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw a continuous line from the intersection points to the image boundaries
                cv.line(cropped_whiteZone, (x_top, y_top), (x_bottom, y_bottom), (0, 255, 0), 2)
                print(x_top, y_top)
                print(x_bottom, y_bottom)



        cv.imshow("RealWhiteCircleZone", cropped_whiteZone)
        # cv.imshow("SolidCircleZone1", frame[148+circles[0][3][1]-100: 148+circles[0][3][1]+100, 174+circles[0][3][0]-100: 174+circles[0][3][0]+100])
        # cv.imshow("SolidCircleZone2", frame[148+circles[0][6][1]-100: 148+circles[0][6][1]+100, 174+circles[0][6][0]-100: 174+circles[0][6][0]+100])
        # cv.imshow("SolidCircleZone3", frame[148+circles[0][12][1]-100: 148+circles[0][12][1]+100, 17q4+circles[0][12][0]-100: 174+circles[0][12][0]+100])
        # cv.imshow("SolidCircleZone4", frame[148+circles[0][8][1]-100: 148+circles[0][8][1]+100, 174+circles[0][8][0]-100: 174+circles[0][8][0]+100])
        # cv.imshow("SolidCircleZone5", frame[148+circles[0][10][1]-100: 148+circles[0][10][1]+100, 174+circles[0][10][0]-100: 174+circles[0][10][0]+100])
        # cv.imshow("SolidCircleZone6", frame[148+circles[0][14][1]-100: 148+circles[0][14][1]+100, 174+circles[0][14][0]-100: 174+circles[0][14][0]+100])
        # cv.imshow("SolidCircleZone7", frame[148+circles[0][9][1]-100: 148+circles[0][9][1]+100, 174+circles[0][9][0]-100: 174+circles[0][9][0]+100])
        
    #     print(f'{i}-1 : {sortedDominantColor[0]}')

    #     minDiffValue = 1000
    #     minDiffPos = -1
    #     for j in range(len(rgbColors)):
    #         pctDiffRed = abs(sortedDominantColor[0][0]-rgbColors[j][0]) / 255
    #         pctDiffGreen = abs(sortedDominantColor[0][1]-rgbColors[j][1]) / 255
    #         pctDiffBlue = abs(sortedDominantColor[0][2]-rgbColors[j][2]) / 255

    #         pctDiffRGB = (pctDiffRed + pctDiffGreen + pctDiffBlue) / 3 * 100

    #         if pctDiffRGB < minDiffValue:
    #             minDiffPos = j
    #             minDiffValue = pctDiffRGB

    #         # print(f'Diff RGB : {pctDiffRGB}') 

    #     # print(f'Less Diff RGB : {minDiffPos}') 
    #     if minDiffPos == 0:
    #         print("Similar to Yellow")
    #     elif minDiffPos == 1:
    #         print("Similar to Blue")
    #     elif minDiffPos == 2:
    #         print("Similar to Red")
    #     elif minDiffPos == 3:
    #         print("Similar to Purple")
    #     elif minDiffPos == 4:
    #         print("Similar to Orange")
    #     elif minDiffPos == 5:
    #         print("Similar to Green")
    #     elif minDiffPos == 6:
    #         print("Similar to Crimson")
    #     elif minDiffPos == 7:
    #         print("Similar to White")
    #     elif minDiffPos == 8:
    #         print("Similar to Black")

    #     cv.imshow("dominant_colors", out)

    # bars = []
    # rgbValues = []

    # for index, row in enumerate(centers):
    #     bar, rgb = createBar(200, 200, row)
    #     bars.append(bar)
    #     rgbValues.append(rgb)

    # imgBar = np.hstack(bars)

    # for index, row in enumerate(rgbValues):
    #     image = cv.putText(imgBar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
    #                         cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    #     print(f'{index + 1}. RGB: {row}')

    # print(f'RGB1TO2 : {abs(rgbValues[0][0]-rgbValues[1][0])+abs(rgbValues[0][1]-rgbValues[1][1])+abs(rgbValues[0][2]-rgbValues[1][2])}')
    # print(f'RGB1TO3 : {abs(rgbValues[0][0]-rgbValues[2][0])+abs(rgbValues[0][1]-rgbValues[2][1])+abs(rgbValues[0][2]-rgbValues[2][2])}')

    # cv.imshow("rgbBar", imgBar)

    # cv.imshow("Frame", frame)
    # cv.imshow("ShowFrame", showFrame)
    cv.imshow("CroppedShowFrame", cropped_Show_image)
    # cv.imshow("cropped", cropped_image)
    # cv.imshow("CircleZoneColor[2]", circleZonesColor[6])
    # cv.imshow("CircleZoneColor[11]", circleZonesColor[8])

    circleZones = []
    
    cv.imshow("circles", frame)
    cv.imshow("cropped", cropped_image)
    circleZonesColor = []

    # cv.imshow("circles", frame)
    # cv.imshow("grayFrame", blurFrame)
    # cv.imshow("Output", outputDrawing)
    # cv.imshow("Detected Circles", cropped_image)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cv.destroyAllWindows()