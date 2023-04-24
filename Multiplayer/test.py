import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    frame = frame[200:1080,0:1920]

    # Convert the frame to grayscale
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurFrame = cv2.GaussianBlur(hsv, (5, 5), 0)

    lower= np.array([145, 110, 205])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(blurFrame, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Extract the largest contour that has more than 10 points
        cue_contour = max([c for c in contours if len(c) > 10], key=cv2.contourArea)

        # Fit a line to the cue contour
        [vx, vy, x, y] = cv2.fitLine(cue_contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Compute the start and end points of the cue line
        start_x = int(x - vx * 2000)
        start_y = int(y - vy * 2000)
        end_x = int(x + vx * 2000)
        end_y = int(y + vy * 2000)
        
        height, width, _ = frame.shape
        if start_x < 0:
            start_x = 0
            start_y = int(y + (start_x - x) * vy / vx)
        elif start_x >= width:
            start_x = width - 1
            start_y = int(y + (start_x - x) * vy / vx)
        if start_y < 0:
            start_y = 0
            start_x = int(x + (start_y - y) * vx / vy)
        elif start_y >= height:
            start_y = height - 1
            start_x = int(x + (start_y - y) * vx / vy)

        if end_x < 0:
            end_x = 0
            end_y = int(y + (end_x - x) * vy / vx)
        elif end_x >= width:
            end_x = width - 1
            end_y = int(y + (end_x - x) * vy / vx)
        if end_y < 0:
            end_y = 0
            end_x = int(x + (end_y - y) * vx / vy)
        elif end_y >= height:
            end_y = height - 1
            end_x = int(x + (end_y - y) * vx / vy)
        # Draw the cue line on the original frame
        print(start_x, start_y,end_x, end_y)
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)

    # Display the frame
    cv2.imshow("Cue Detection", frame)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
