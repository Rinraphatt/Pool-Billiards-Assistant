import cv2
import numpy as np

# Callback function for trackbar
def nothing(x):
    pass

# Create a window
cv2.namedWindow('Canny Edge Detection')

# Create trackbars for threshold values
cv2.createTrackbar('min', 'Canny Edge Detection', 0, 255, nothing)
cv2.createTrackbar('max', 'Canny Edge Detection', 0, 255, nothing)

# Open the default camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurFrame = cv2.GaussianBlur(hsv, (7, 7), 0)
    # Define a cue white color threshold
    lower_cue = np.array([145, 120, 140])
    upper_cue = np.array([170, 255, 255])
    mask = cv2.inRange(blurFrame, lower_cue, upper_cue)
    kernel = np.ones((3,3),np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # dilate->erode
    mask = cv2.dilate(mask_closing,kernel,iterations = 1)
    cv2.imshow("1",mask)
    # Get current threshold values
    min_val = cv2.getTrackbarPos('min', 'Canny Edge Detection')
    max_val = cv2.getTrackbarPos('max', 'Canny Edge Detection')

    # Apply Canny edge detection
    edges = cv2.Canny(mask, min_val, max_val)
    kernel = np.ones((3,3),np.uint8)
    mask_closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) # dilate->erode
    # Show image with edges
    cv2.imshow('Canny Edge Detection', mask_closing)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
