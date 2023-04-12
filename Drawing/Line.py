import cv2
import numpy as np

def on_trackbar(val):
    # Get current trackbar values
    threshold = cv2.getTrackbarPos('Threshold', 'Hough Lines')
    minLineLength = cv2.getTrackbarPos('Min Line Length', 'Hough Lines')
    maxLineGap = cv2.getTrackbarPos('Max Line Gap', 'Hough Lines')

    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength, maxLineGap)

    # Draw lines on the original image
    img_copy = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the image with detected lines
    cv2.imshow('Hough Lines', img_copy)

# Load an image
img = cv2.imread('./pics/new13.jpg')

# Convert to grayscale and apply Canny edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Create a window for displaying the image with detected lines
cv2.namedWindow('Hough Lines')

# Create trackbars for adjusting HoughLinesP parameters
cv2.createTrackbar('Threshold', 'Hough Lines', 50, 200, on_trackbar)
cv2.createTrackbar('Min Line Length', 'Hough Lines', 50, 200, on_trackbar)
cv2.createTrackbar('Max Line Gap', 'Hough Lines', 10, 100, on_trackbar)

# Call the trackbar function once to display the initial image with default parameters
on_trackbar(0)

# Wait for a key event
cv2.waitKey(0)

# Destroy the window