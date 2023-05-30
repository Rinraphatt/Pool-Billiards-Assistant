import cv2
import numpy as np

# Define a function to apply the colormap to the grayscale image and display the result
def adjust_tone(value):
    # Apply the selected colormap to the grayscale image
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET + value)

    # Show the colored image
    cv2.imshow('Colored1', colored)

# Create a VideoCapture object to capture the video stream from the webcam
cap = cv2.VideoCapture(0)

# Create a window to display the images and trackbar
cv2.namedWindow('Colored')
cv2.createTrackbar('Tone', 'Colored', 0, 255, adjust_tone)
cv2.createTrackbar('Tone', 'Colored', 0, 255, adjust_tone)

# Continuously capture frames from the webcam and display them with the adjusted tone
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the selected colormap to the grayscale image
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Show the colored image
    cv2.imshow('Colored', colored)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
