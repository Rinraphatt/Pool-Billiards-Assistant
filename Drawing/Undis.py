import cv2
import numpy as np

# Load the camera matrix and distortion coefficients from the calibration file
mtx = np.loadtxt('../arUco/calib_data/camera_matrix.txt')
dist = np.loadtxt('../arUco/calib_data/dist_coeffs.txt')
print("Loaded")
# Load the image
img = cv2.imread('../pics/pic15.jpg')
img = cv2.resize(img, (1920, 1080))
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# Load the image to be projected
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Show the original and undistorted images side by side
cv2.imshow('Original', img)
cv2.imshow('Undistorted', dst)
cv2.imwrite('../pics/Undist.jpg',dst)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
