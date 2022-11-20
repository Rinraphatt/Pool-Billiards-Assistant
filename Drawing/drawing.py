import numpy as np
import cv2 as cv

img = np.zeros((1080,1920,3), np.uint8)

while True:

    cv.line(img,(0,0),(1920,1080),(255,0,0),5)
    
    cv.circle(img,(447,63), 63, (255,0,0), -1)

    cv.imshow("Test Drawing", img)
    
    if cv.waitKey(1) & 0xFF == ord('q'): break

cv.destroyAllWindows()