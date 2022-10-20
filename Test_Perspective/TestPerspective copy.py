import numpy as np
import cv2




frame = cv2.imread("Test_Perspective/new.jpg")

while True:
    tl = (232 ,219)
    bl = (236 ,1235)
    tr = (2325, 207)
    br = (2324 ,1234)
    cv2.circle(frame, tl, 2, (0, 0, 255), -1)
    cv2.circle(frame, bl, 2, (0, 0, 255), -1)
    cv2.circle(frame, tr, 2, (0, 0, 255), -1)
    cv2.circle(frame, br, 2, (0, 0, 255), -1)
    cv2.imshow("Test", frame)
    cv2.waitKey(0)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 900], [1800, 0], [1800, 900]])
        
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    tansformed_frame = cv2.warpPerspective(frame, matrix, (1800, 900))
    cv2.imshow("Test", frame)
    cv2.imshow("Test_Perspectice", tansformed_frame)
    cv2.imwrite("crop.jpg",tansformed_frame)
    if cv2.waitKey(1) == 27:
        break