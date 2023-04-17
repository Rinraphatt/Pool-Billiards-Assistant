import cv2
import numpy as np
width = 1920
height = 1080
# Callback function for trackbar
def nothing(x):
    pass

# Load the video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# Create a window for trackbars
cv2.namedWindow('Hough Circle Detection')

# Create trackbars for parameters
cv2.createTrackbar('minRadius', 'Hough Circle Detection', 40, 100, nothing)
cv2.createTrackbar('maxRadius', 'Hough Circle Detection', 40, 100, nothing)
cv2.createTrackbar('param1', 'Hough Circle Detection', 1, 100, nothing)
cv2.createTrackbar('param2', 'Hough Circle Detection', 1, 100, nothing)

while(cap.isOpened()):
    ret, frame = cap.read()
    tl = (251, 180)
    bl = (179, 927)
    tr = (1697, 197)
    br = (1749, 942)
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
    print(matrix)
    # Compute the perspective transform M
    frame = cv2.warpPerspective(frame, matrix, (width, height))
    blurFrame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsvFrame = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([45,80,70])
    upper_green = np.array([75,255,255])

    mask = cv2.inRange(hsvFrame, lower_green, upper_green)
    if ret:
        # Get current values of trackbars
        minRadius = cv2.getTrackbarPos('minRadius', 'Hough Circle Detection')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'Hough Circle Detection')
        param1 = cv2.getTrackbarPos('param1', 'Hough Circle Detection')
        param2 = cv2.getTrackbarPos('param2', 'Hough Circle Detection')

        # Convert the frame to grayscale
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Hough Circle detection
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.4, 30, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        # Draw detected circles on the original frame
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

        # Display the original frame and the detected circles
        cv2.imshow('Hough Circle Detection', frame)
        cv2.imshow('Hough Circle Detection1', mask)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()