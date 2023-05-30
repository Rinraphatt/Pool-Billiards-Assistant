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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # Get current threshold values
    min_val = cv2.getTrackbarPos('min', 'Canny Edge Detection')
    max_val = cv2.getTrackbarPos('max', 'Canny Edge Detection')

    # Apply Canny edge detection
    edges = cv2.Canny(gray, min_val, max_val)
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
