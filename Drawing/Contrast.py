import cv2
import numpy as np

def adjust_contrast_and_brightness(value):
    # Retrieve the current trackbar values for contrast and brightness
    alpha = cv2.getTrackbarPos("Contrast", "Video Feed") / 10.0
    beta = cv2.getTrackbarPos("Brightness", "Video Feed")

    # Apply the contrast and brightness adjustments to the video feed
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Show the adjusted video feed
    cv2.imshow("Video Feed1", adjusted)

# Create a window to display the video feed
cv2.namedWindow("Video Feed")

# Create trackbars to adjust the contrast and brightness of the video feed
cv2.createTrackbar("Contrast", "Video Feed", 10, 20, adjust_contrast_and_brightness)
cv2.createTrackbar("Brightness", "Video Feed", 0, 100, adjust_contrast_and_brightness)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Continuously read frames from the webcam and display them with the adjusted contrast and brightness
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If the frame was not successfully read, break the loop
    if not ret:
        break

    # Apply the initial contrast and brightness adjustments to the video feed
    adjusted = cv2.convertScaleAbs(frame, alpha=1.0, beta=0)

    # Show the initial video feed
    cv2.imshow("Video Feed", adjusted)

    # Wait for a key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
