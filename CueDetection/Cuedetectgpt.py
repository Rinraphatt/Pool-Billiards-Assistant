import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture('./videos/Test.mp4')

# Loop through the video frames
while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(gray, 130, 255)

    # Apply the Hough Line Transform to detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, minLineLength=150, maxLineGap=10)

    # Draw the detected line segments on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Pool table with cue detected', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()