import numpy as np
import cv2
def adjustThreshold():
    print("val")
vidcap = cv2.VideoCapture("Test_Perspective/newVid.mp4")
succuess, img = vidcap.read()
width = 1280
heigth = 720
cv2.namedWindow("Test_Perspectice")
cv2.createTrackbar("minThreshold","Test_Perspectice",30,300,adjustThreshold)
cv2.createTrackbar("maxThreshold","Test_Perspectice",70,300,adjustThreshold)
cv2.createTrackbar("minR","Test_Perspectice",2,300,adjustThreshold)
cv2.createTrackbar("maxR","Test_Perspectice",10,300,adjustThreshold)

while succuess:
    succuess, img = vidcap.read()
    frame = img

    if succuess == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))

        mint  = cv2.getTrackbarPos("minThreshold","Test_Perspectice")
        maxt  = cv2.getTrackbarPos("maxThreshold","Test_Perspectice")
        minr  = cv2.getTrackbarPos("minR","Test_Perspectice")
        maxr  = cv2.getTrackbarPos("maxR","Test_Perspectice")
        print(mint,maxt)
        detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 30, param1 = int(maxt),
               param2 = int(mint), minRadius = minr, maxRadius = maxr )
        # Draw circles that are detected.

    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
    
            # Draw the circumference of the circle.
            cv2.circle(frame, (a, b), r, (0, 255, 0), 2)
    
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)
  
        tl = (177 ,159)
        bl = (180 ,922)
        tr = (1741 ,155)
        br = (1742 ,925)
        cv2.circle(frame, tl, 2, (0, 0, 255), -1)
        cv2.circle(frame, bl, 2, (0, 0, 255), -1)
        cv2.circle(frame, tr, 2, (0, 0, 255), -1)
        cv2.circle(frame, br, 2, (0, 0, 255), -1)
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, heigth], [width, 0], [width, heigth]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Compute the perspective transform M
        tansformed_frame = cv2.warpPerspective(frame, matrix, (width, heigth))

        cv2.imshow("Test", frame)
        cv2.imshow("Test_Perspectice", tansformed_frame)
        if cv2.waitKey(1) == 27:
            break
    else:
       break
