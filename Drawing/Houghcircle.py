import cv2
import numpy as np
width = 1920
height = 1080
# Callback function for trackbar
def nothing(x):
    pass

def createTable():
    h,w = 880,1920
    frame = np.zeros((h, w, 1), np.uint8)
    pocket_point = [(0,0),(int(w/2),-30),(w,0),(0,h),(int(w/2),h+30),(w,h)]
    for i in pocket_point:
        cv2.circle(frame, i , 50, (255, 255, 255), -1)
    return frame

# Load the video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# Create a window for trackbars

cv2.namedWindow('Hough Circle Detection')
cv2.namedWindow('Hough Circle Detection',cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Hough Circle Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Create trackbars for parameters
cv2.createTrackbar('minRadius', 'Hough Circle Detection', 40, 100, nothing)
cv2.createTrackbar('maxRadius', 'Hough Circle Detection', 40, 100, nothing)
cv2.createTrackbar('param1', 'Hough Circle Detection', 1, 100, nothing)
cv2.createTrackbar('param2', 'Hough Circle Detection', 1, 100, nothing)

while(cap.isOpened()):
    ret, frame = cap.read()
    tl = (245 ,10)
    bl = (180 ,900)
    tr = (1717 ,22)
    br = (1760 ,930)
    cv2.circle(frame, tl, 3, (0, 0, 255), -1)
    cv2.circle(frame, bl, 3, (0, 0, 255), -1)
    cv2.circle(frame, tr, 3, (0, 0, 255), -1)
    cv2.circle(frame, br, 3, (0, 0, 255), -1)
    cv2.line(frame, tl, bl, (0, 255, 0), 2)
    cv2.line(frame, bl, br, (0, 255, 0), 2)
    cv2.line(frame, br, tr, (0, 255, 0), 2)
    cv2.line(frame, tl, tr, (0, 255, 0), 2)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    frame = cv2.warpPerspective(frame, matrix, (width, height))
    frame = frame[200:1080,0:1920]
    blurFrame = cv2.GaussianBlur(frame, (7, 7), 0)
    # Apply Canny edge detection
    gray = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 20, 50)
    # kernel = np.ones((3,3),np.uint8)
    # mask_closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) # dilate->erode

    hsv = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)
    # Set minimum and max HSV values to display
    lower = np.array([50, 0, 0])
    upper = np.array([90, 255, 255])
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # dilate->erode
    mask = cv2.dilate(mask_closing,kernel,iterations = 1)
    inv_mask = cv2.bitwise_not(mask)
    output = cv2.bitwise_and(frame,frame, mask= inv_mask)
    black = createTable()
    ret,thresh1 = cv2.threshold(black,127,255,cv2.THRESH_BINARY)
    mask = cv2.inRange(hsvFrame, lower_bg, upper_bg)
    blurFrame = cv2.GaussianBlur(mask, (7,7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # dilate->erode
    mask = cv2.dilate(mask_closing,kernel,iterations = 1)
    inv_mask = cv2.bitwise_not(mask)
    table_mask = inv_mask | thresh1

    if ret:
        # Get current values of trackbars
        minRadius = cv2.getTrackbarPos('minRadius', 'Hough Circle Detection')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'Hough Circle Detection')
        param1 = cv2.getTrackbarPos('param1', 'Hough Circle Detection')
        param2 = cv2.getTrackbarPos('param2', 'Hough Circle Detection')

        # Convert the frame to grayscale
        #gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        #Apply Hough Circle detection
        circles = cv2.HoughCircles(inv_mask, cv2.HOUGH_GRADIENT, 1, 30, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        # Draw detected circles on the original frame
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

        # # Filter the contours based on their area, circularity, and aspect ratio
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     perimeter = cv2.arcLength(contour, True)
        #     #print(perimeter)
        #     if perimeter <= 0 :
        #         perimeter = 0.001
        #     circularity = 4 * 3.14159 * area / (perimeter * perimeter)
        #     x, y, w, h = cv2.boundingRect(contour)
        #     aspect_ratio = float(w) / h
        #     if area > 200 and circularity > 0.6 and aspect_ratio < 1.2:
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the original frame and the detected circles
        cv2.imshow('Hough Circle Detection', frame)
        cv2.imshow('Hough Circle Detection2', output)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()