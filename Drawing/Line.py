import cv2
import numpy as np

def on_trackbar(val):
    # Get current trackbar values
    threshold = cv2.getTrackbarPos('Threshold', 'Hough Lines')
    minLineLength = cv2.getTrackbarPos('Min Line Length', 'Hough Lines')
    maxLineGap = cv2.getTrackbarPos('Max Line Gap', 'Hough Lines')

    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(mask_closing, 1, np.pi/180, threshold, minLineLength, maxLineGap)

    # Draw lines on the original image
    img_copy = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the image with detected lines
    cv2.imshow('Hough Lines', img_copy)

# Create a window for displaying the image with detected lines
cv2.namedWindow('Hough Lines')

# Create trackbars for adjusting HoughLinesP parameters
cv2.createTrackbar('Threshold', 'Hough Lines', 50, 200, on_trackbar)
cv2.createTrackbar('Min Line Length', 'Hough Lines', 50, 200, on_trackbar)
cv2.createTrackbar('Max Line Gap', 'Hough Lines', 10, 100, on_trackbar)

# Initialize the webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurFrame = cv2.GaussianBlur(hsv, (7, 7), 0)
    # Define a cue white color threshold
    lower_cue = np.array([145, 120, 140])
    upper_cue = np.array([170, 255, 255])
    mask = cv2.inRange(blurFrame, lower_cue, upper_cue)
    kernel = np.ones((3,3),np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # dilate->erode
    mask = cv2.dilate(mask_closing,kernel,iterations = 1)
    # Apply Canny edge detection
    edges = cv2.Canny(mask, 180, 255)
    mask_closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) # dilate->erode
    cv2.imshow("1",mask_closing)
    # Call the trackbar function to detect and display lines
    on_trackbar(0)

    # Check for key event to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy the window
cap.release()
cv2.destroyAllWindows()
