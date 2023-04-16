import cv2
import cv2.aruco as aruco

# Define the size of the ArUco markers
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
marker_size = 0.04 # meters

# Generate the Charuco board
board = aruco.CharucoBoard_create(7, 5, marker_size, 0.03, aruco_dict)

# Draw the Charuco board and resize it to fit an A4 paper
img = board.draw((1920, 1080))
a4_width = 21.0 # cm
a4_height = 29.7 # cm
a4_size = (int(a4_width * 96), int(a4_height * 96)) # calculate size in pixels at 96 dpi
img_resized = cv2.resize(img, a4_size)

# Save the Charuco board image to a file
cv2.imwrite("charuco_board.png", img)

# Display the Charuco board image
cv2.imshow("Charuco board", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
