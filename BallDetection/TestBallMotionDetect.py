
import cv2
import numpy as np

# cam = cv.VideoCapture(0)

cam = cv2.VideoCapture('../videos/Level1_White1.mp4')
width = 1920
height = 1080

bgFrame = None

while True:
    ret, frame = cam.read()
    ret2, frame2 = cam.read()

    if not ret:
        break

    # Perspective Transform
    tl = (212, 180)
    bl = (159, 923)
    tr = (1696, 178)
    br = (1750, 925)
    cv2.circle(frame, tl, 3, (0, 0, 255), -1)
    cv2.circle(frame, bl, 3, (0, 0, 255), -1)
    cv2.circle(frame, tr, 3, (0, 0, 255), -1)
    cv2.circle(frame, br, 3, (0, 0, 255), -1)
    cv2.line(frame, tl, bl, (0, 255, 0), 2)
    cv2.line(frame, bl, br, (0, 255, 0), 2)
    cv2.line(frame, br, tr, (0, 255, 0), 2)
    cv2.line(frame, tl, tr, (0, 255, 0), 2)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    frame = cv2.warpPerspective(frame, matrix, (width, height))
    showFrame = cv2.warpPerspective(frame2, matrix, (width, height))

    # Check if there are motion or not
    motionGrayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motionBlurFrame = cv2.GaussianBlur(motionGrayFrame, (35, 35), 0)

    if bgFrame is None:
        bgFrame = motionBlurFrame

    frameDelta = cv2.absdiff(bgFrame, motionBlurFrame)
    _, thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)

    n_white_pix = np.sum(thresh == 255)

    print("n_white_pix = ", n_white_pix)

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

    bgFrame = motionBlurFrame

    cv2.imshow("Real", frame)
    cv2.imshow("Blur", motionBlurFrame)
    cv2.imshow("AbsoluteDiff", frameDelta)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
