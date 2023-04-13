import cv2
import numpy as np
import time

# Define the lower and upper bounds of the skin color in the HSV color space
lower_skin = np.array([0, 21, 180], dtype=np.uint8)
upper_skin = np.array([179, 50, 255], dtype=np.uint8)

# Define the rectangular zone
x1, y1 = 200, 200  # top-left corner
x2, y2 = 300, 300  # bottom-right corner

# Start capturing the video feed from the webcam
cap = cv2.VideoCapture(0)

# Initialize the timer
start_time = None

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply the skin color segmentation to the HSV frame
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to remove noise and fill holes in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find the contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        # Find the centroid of the largest contour
        M = cv2.moments(largest_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # Check if the centroid is inside the rectangular zone
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= 2:
                print('Hand stayed in the zone for 3 seconds!')
        else:
            start_time = None
        # Draw a rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw the rectangular zone on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('Hand Zone', frame)

    # Check for user input to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
