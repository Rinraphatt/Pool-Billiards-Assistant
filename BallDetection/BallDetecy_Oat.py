import numpy as np
import cv2

vidcap = cv2.VideoCapture('./videos/Test.mp4')
succuess, img = vidcap.read()
width = 1280
heigth = 720
while succuess:
    succuess, img = vidcap.read()
    frame = img

    tl = (177, 159)
    bl = (180, 922)
    tr = (1741, 155)
    br = (1742, 925)

    cv2.circle(frame, tl, 3, (0, 0, 255), -1)
    cv2.circle(frame, bl, 3, (0, 0, 255), -1)
    cv2.circle(frame, tr, 3, (0, 0, 255), -1)
    cv2.circle(frame, br, 3, (0, 0, 255), -1)

    cv2.line(frame, tl, bl, (0, 255, 0), 2)
    cv2.line(frame, bl, br, (0, 255, 0), 2)
    cv2.line(frame, br, tr, (0, 255, 0), 2)
    cv2.line(frame, tl, tr, (0, 255, 0), 2)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, heigth], [width, 0], [width, heigth]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Compute the perspective transform M
    tansformed_frame = cv2.warpPerspective(frame, matrix, (width, heigth))

    grayFrame = cv2.cvtColor(tansformed_frame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame, (7, 7), 0)
    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                               param1=80, param2=30, minRadius=22, maxRadius=35)
    circleZones = []
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(tansformed_frame, cv2.COLOR_BGR2HSV)

    # # Define a white color threshold
    # lower_white = np.array([0, 0, 130])
    # upper_white = np.array([179, 50, 255])

    # Define a cue white color threshold
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([179, 80, 255])

    # Apply the white color threshold to extract white regions
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours in the white regions
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(tansformed_frame, (x, y), r, (0, 255, 0), 2)

    if contours:
        # Find the contour with the largest area
        max_contour = max(contours, key=cv2.contourArea)

        # Fit a circle to the largest contour
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw a circle around the largest contour
        cv2.circle(tansformed_frame, center, radius, (0, 0, 255), 2)
        # Draw a rec around the largest contour
        # x, y, w, h = cv2.boundingRect(max_contour)
        # cv2.rectangle(tansformed_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Crop the image using the circle
        mask = np.zeros_like(tansformed_frame)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        crop_frame = cv2.bitwise_and(tansformed_frame, mask)
        # crop_frame = tansformed_frame[y:y+h, x:x+w]

       

    cv2.imshow("Circle and White Ball Detection", tansformed_frame)
    cv2.imshow("Crop", crop_frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close all windows
cap.release()
cv2.destroyAllWindows()
