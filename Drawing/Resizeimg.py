import cv2
import numpy as np
import time
import math

width = 1920
height = 1080
output_width = 1920
output_height = 1080
# Define the lower and upper bounds of the skin color in the HSV color space
lower_skin = np.array([0, 21, 180], dtype=np.uint8)
upper_skin = np.array([179, 250, 255], dtype=np.uint8)

# Define the rectangular zone
x1, y1 = 200, 200  # top-left corner
x2, y2 = 300, 300  # bottom-right corner

# Start capturing the video feed from the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# Initialize the timer
start_time = None
tl = (251, 170)
bl = (179, 927)
tr = (1697, 187)
br = (1749, 942)

def max_width_height(p1, p2, p3, p4):
    s1 = abs(p1[0]-p3[0])
    s2 = abs(p2[0]-p4[0])
    s3 = abs(p1[1]-p2[1])
    s4 = abs(p3[1]-p4[1])
    width = max(s1,s2)
    height = max(s3,s4)

    return (width,height)


def are_rectangles_overlapping(rect1, rect2):
    """
    Checks if two rectangles are overlapping
    rect1, rect2: tuples of 4 integers representing (x, y, width, height) of each rectangle
    returns: True if the rectangles are overlapping, False otherwise
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True
    


# Set the new resolution
new_width, new_height = 1920, 1080
print(max_width_height(tl,bl,tr,br))
output_width , output_height = max_width_height(tl,bl,tr,br)
while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, output_height], [output_width, 0], [output_width, output_height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    frame = cv2.warpPerspective(frame, matrix, (output_width ,output_height))
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply the skin color segmentation to the HSV frame
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to remove noise and fill holes in the mask
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    cv2.imshow('Hand Zone1', mask)

    # Find the contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw a rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rect1 = x,y,w,h
        rect2 = 200,200,100,100
        # Draw the rectangular zone on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if are_rectangles_overlapping(rect1,rect2) == True :
            
            if start_time is None:
                start_time = time.time()

            elif time.time() - start_time >= 2:
                print('Hand stayed in the zone for 3 seconds!')
        else:
            start_time = None
    # Display the frame

    
   
    cv2.imshow('Hand Zone', frame)
    # Get the dimensions of the original image
    height, width, channels = frame.shape
    # Determine the scale factor to scale up the image
    scale_factor = min(new_width/width, new_height/height)

    # Resize the image
    resized_img = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    # Create a new blank image with the new resolution
    new_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    print(resized_img.shape[:2])
    # Calculate the position of the resized image in the new image
    # x_offset = int((new_width - resized_img.shape[1])/2)
    # y_offset = int((new_height - resized_img.shape[0])/2)
    # Paste the resized image into the new image
    new_img[1080-(resized_img.shape[0]):, 0:1920] = resized_img
    
    # Display the new image
    cv2.namedWindow('New Image',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('New Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('New Image', new_img)
    # Check for user input to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
