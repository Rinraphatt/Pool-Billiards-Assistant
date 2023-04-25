import numpy as np
import cv2

# Create a black image
mac = np.zeros((880, 1920, 3), np.uint8)

# Draw a circle on the image
cv2.rectangle(mac, (0, 0), (511, 511), (0, 0, 0), -1)
cv2.circle(mac, (256, 256), 100, (0, 255, 0), -1)

# Load your source video
vidcap = cv2.VideoCapture(0)
width = 1920
height = 1080
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

mtx = np.loadtxt('../arUco/calib_data/camera_matrix.txt')
dist = np.loadtxt('../arUco/calib_data/dist_coeffs.txt')
print("Loaded")

while True:
    # Read a frame from the video
    ret, frame = vidcap.read()

    if ret:
        frame = cv2.undistort(frame, mtx, dist)
        tl = (245 ,10)
        bl = (180 ,900)
        tr = (1717 ,22)
        br = (1760 ,930)

        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Compute the perspective transform M
        tansformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
        detectFrame = tansformed_frame.copy()

        tansformed_frame[200:1080,0:1920] = mac

        # Show the resulting frame\
        cv2.namedWindow('Frame',cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Frame', tansformed_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close all windows
vidcap.release()
cv2.destroyAllWindows()

