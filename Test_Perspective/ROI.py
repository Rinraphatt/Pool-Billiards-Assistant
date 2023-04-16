import cv2
import numpy as np

# Load the image
img = cv2.imread('./pics/1.jpg')
img = cv2.resize(img, (1920, 1080))

# Initialize the list of points for the rectangle
rect_points = []

# Define a mouse callback function
def print_pixel_position(event, x, y, flags, param):
    global rect_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the point to the rectangle points list
        rect_points.append((x, y))

        # If we have 4 points, draw the rectangle
        if len(rect_points) == 4:
            # Draw the rectangle
            rect = np.array(rect_points, np.int32)
            rect = rect.reshape((-1,1,2))
            cv2.polylines(img, [rect], True, (0,255,0), 2)

            # Clear the rectangle points list
            rect_points.clear()

        # Display the image
        cv2.imshow('Image', img)

def clear_polylines():
    global rect_points
    rect_points.clear()
    cv2.imshow('Image', img)

# Display the image
cv2.imshow('Image', img)

# Set the mouse callback function for the window
cv2.setMouseCallback('Image', print_pixel_position)

# Main loop
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        clear_polylines()
        print("Clleeat")
    elif key == 27:
        break

cv2.destroyAllWindows()
