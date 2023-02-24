import numpy as np
import cv2

vidcap = cv2.VideoCapture(0)
succuess, img = vidcap.read()
width = 1280
heigth = 720

print(succuess)

while succuess:
    succuess, img = vidcap.read()
    frame = img

    tl = (177 ,159)
    bl = (180 ,922)
    tr = (1741 ,155)
    br = (1742 ,925)
    cv2.circle(frame, tl, 3, (0, 0, 255), -1)
    cv2.circle(frame, bl, 3, (0, 0, 255), -1)
    cv2.circle(frame, tr, 3, (0, 0, 255), -1)
    cv2.circle(frame, br, 3, (0, 0, 255), -1)
    cv2.line(frame, tl, bl, (0, 255, 0), 2)
    cv2.line(frame, bl, br, (0, 255, 0), 2)
    cv2.line(frame, br, tr, (0, 255, 0), 2)
    cv2.line(frame, tl, tr, (0, 255, 0), 2)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, heigth], [width, 0], [width, heigth]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    tansformed_frame = cv2.warpPerspective(frame, matrix, (width, heigth))
    cv2.imshow("Test_Perspectice", tansformed_frame)
    cv2.imshow("Test", frame)
    
    if cv2.waitKey(1) == 27:
        break
