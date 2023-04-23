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

    lower= np.array([145, 0, 136])
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
        start_x = int(x - vx * 1000)
        start_y = int(y - vy * 1000)
        end_x = int(x + vx * 1000)
        end_y = int(y + vy * 1000)

        # # Limit the line to the edges of the screen
        # height, width, _ = frame.shape
        # if start_y < 0:
        #     start_x = int(x - (vy/vx) * (start_y - y))
        #     start_y = 0
        # if end_y > height:
        #     end_x = int(x + (vy/vx) * (height - end_y))
        #     end_y = height
        # if start_x < 0:
        #     start_y = int(y - (vx/vy) * (start_x - x))
        #     start_x = 0 
        # if end_x > width:
        #     end_y = int(y + (vx/vy) * (width - end_x))
        #     end_x = width

        # Draw the cue line on the original frame
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
