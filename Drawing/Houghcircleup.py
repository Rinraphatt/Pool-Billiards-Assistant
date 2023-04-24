import cv2
import numpy as np
# Create a video capture object
width = 1920
height = 1080
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# Create a window with trackbars to adjust parameters
cv2.namedWindow('Controls')
cv2.createTrackbar('MinRadius', 'Controls', 20, 100, lambda x: None)
cv2.createTrackbar('MaxRadius', 'Controls', 50, 200, lambda x: None)

tl = (245 ,10)
bl = (180 ,900)
tr = (1717 ,22)
br = (1760 ,930)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    frame = cv2.warpPerspective(frame, matrix, (width, height))
    frame = frame[200:1080,0:1920]
    # blurFrame = cv2.GaussianBlur(frame, (5, 5), 0)
    # hsvFrame = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([45,80,70])
    # upper_green = np.array([75,255,255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurFrame = cv2.GaussianBlur(hsv, (5, 5), 0)
    lower= np.array([40, 50, 85])
    upper = np.array([75, 240, 255])
    mask = cv2.inRange(blurFrame, lower, upper)
    # kernel = np.ones((1, 1), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    inv_mask = cv2.bitwise_not(mask)
    output = cv2.bitwise_and(frame,frame, mask= inv_mask)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the circles in the grayscale image
    min_radius = cv2.getTrackbarPos('MinRadius', 'Controls')
    max_radius = cv2.getTrackbarPos('MaxRadius', 'Controls')
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=min_radius,
                               maxRadius=max_radius)

    # Filter the detected circles and fit an ellipse to the half circle contour
    if circles is not None:
        for circle in circles[0]:
            x, y, r = circle
            if y > r and y < gray.shape[0] - r and x > r and x < gray.shape[1] - r:
                half_circle = gray[y-r:y+r, x-r:x+r]
                ellipse = cv2.fitEllipse(cv2.findNonZero(half_circle))

                # Check the aspect ratio of the ellipse to ensure that it's a half circle
                aspect_ratio = ellipse[1][0] / ellipse[1][1]
                if aspect_ratio < 0.7:
                    cv2.ellipse(frame, (int(x), int(y)), (int(r), int(r)), ellipse[2], 90, 270, (0, 255, 0), 2)

    # Show the original frame with detected half circles
    cv2.imshow('Controls', frame)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy the windows
cap.release()
cv2.destroyAllWindows()
