
import cv2 as cv
import numpy as np

# cam = cv.VideoCapture(0)

cam = cv.VideoCapture('../videos/new1080.mp4')

cv.namedWindow("Python Webcam Screenshot App")

bgFrame = None

while True:
    ret, frame = cam.read()

    if not ret: break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (35,35), 0)

    if bgFrame is None:
        bgFrame = blurFrame

    frameDelta = cv.absdiff(bgFrame, blurFrame)
    _,thresh = cv.threshold(frameDelta, 10, 255, cv.THRESH_BINARY)

    n_white_pix = np.sum(thresh == 255)
    # print('Number of white pixels:', n_white_pix)

    if n_white_pix <= 100:
        print("IDLE")
    else:
        print("MOVING")

    # thresh = cv.dilate(thresh, None, iterations = 6)

    # contours, hirarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # for cnt in contours:
    #     x, y, w, h = cv.boundingRect(cnt)

    #     cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255),2)
    #     cv.putText(frame, 'Motion Detected', (x,y-3), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

    bgFrame = blurFrame

    cv.imshow("Real", frame)
    cv.imshow("Blur", blurFrame)
    cv.imshow("AbsoluteDiff", frameDelta)
    cv.imshow("Threshold", thresh)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cam.release()
cv.destroyAllWindows()