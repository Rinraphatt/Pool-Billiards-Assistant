import cv2 as cv
import numpy as np

height=1080 
width=1920

# cam = cv.VideoCapture('../Test_Perspective/newVid.mp4')
cam = cv.VideoCapture(0, cv.CAP_DSHOW)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv.CAP_PROP_FRAME_WIDTH, width)
success, img = cam.read()

print(success)

while True:
    success, img = cam.read()
    cv.imshow("cameraTest", img)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cv.destroyAllWindows()
cam.release()