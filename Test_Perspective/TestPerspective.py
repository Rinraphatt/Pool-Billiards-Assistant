import numpy as np
import cv2

vidcap = cv2.VideoCapture("myVid.mp4")
succuess, img = vidcap.read()

while succuess:
    succuess, img = vidcap.read()
    frame = cv2.resize(img, (640, 480))

    tl = (82, 69)
    bl = (76, 430)
    tr = (568, 63)
    br = (578, 430)
    cv2.circle(frame, tl, 2, (0, 0, 255), -1)
    cv2.circle(frame, bl, 2, (0, 0, 255), -1)
    cv2.circle(frame, tr, 2, (0, 0, 255), -1)
    cv2.circle(frame, br, 2, (0, 0, 255), -1)
    cv2.imshow("Test", frame)
    cv2.waitKey(0)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    tansformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    cv2.imshow("Test", frame)
    cv2.imshow("Test_Perspectice", tansformed_frame)
    if cv2.waitKey(1) == 27:
        break
