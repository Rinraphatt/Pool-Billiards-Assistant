import cv2
import numpy as np

# Load the source image
img = cv2.imread('./pics/pic8.jpg')

# Define the corners of the ROI in the source image
src_pts = np.float32([[100, 100], [300, 100], [300, 300], [100, 300]])

# Define the corresponding points in the destination image
dst_pts = np.float32([[200, 100], [400, 100], [400, 300], [200, 300]])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective transform to the source image
warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Warped Image', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()