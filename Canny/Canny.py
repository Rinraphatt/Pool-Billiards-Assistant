import cv2
import numpy as np

# Callback function for trackbar
def nothing(x):
    pass

# Load image
img = cv2.imread('./pics/new13.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (1280, 720))
# Create a window
cv2.namedWindow('Canny Edge Detection')

# Create trackbars for threshold values
cv2.createTrackbar('min', 'Canny Edge Detection', 0, 255, nothing)
cv2.createTrackbar('max', 'Canny Edge Detection', 0, 255, nothing)

while True:
    # Get current threshold values
    min_val = cv2.getTrackbarPos('min', 'Canny Edge Detection')
    max_val = cv2.getTrackbarPos('max', 'Canny Edge Detection')

    # Apply Canny edge detection
    edges = cv2.Canny(img, min_val, max_val)

    # Show image with edges
    cv2.imshow('Canny Edge Detection', edges)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()