import cv2
import numpy as np
calib_data_path = "./arUco/calib_data/MultiMatrix1.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
# Load the image
img = cv2.imread('./pics/pic12.jpg')
img = cv2.resize(img, (1920, 1080))
img = cv2.undistort(img, cam_mat, dist_coef)
# Define a mouse callback function
def print_pixel_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the pixel position on the image
        cv2.circle(img,  (x, y), 3, (0, 0, 255), -1)
        cv2.putText(img, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Image', img)

# Display the image
cv2.imshow('Image', img)

# Set the mouse callback function for the window
cv2.setMouseCallback('Image', print_pixel_position)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
