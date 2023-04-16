import numpy as np
import cv2

vidcap = cv2.VideoCapture(0)
width = 1920
height = 1080
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
calib_data_path = "./arUco/calib_data/MultiMatrix1.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
while True:
    succuess, img = vidcap.read()
    frame = img
    # Undistort the frame
    frame = cv2.undistort(frame, cam_mat, dist_coef)
    #mac = cv2.resize(mac, (1895, 880))
    tl = (261, 62)
    bl = (179, 927)
    tr = (1693, 72)
    br = (1749, 942)
    # tl = (251, 180)
    # bl = (179, 927)
    # tr = (1697, 197)
    # br = (1749, 942)
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
    tansformed_frame = cv2.warpPerspective(frame, matrix, (width, height))

    roi = tansformed_frame[160:1080, 0:1920]  # y1:y2, x1:x2

    # Define the width and height of the ROI
    roi_width = roi.shape[1]
    roi_height = roi.shape[0]

    # Load the image you want to put on the ROI
    overlay = cv2.imread('iMac - 30.png')
    # Resize the overlay to fit the ROI
    overlay_resized = cv2.resize(overlay, (roi_width, roi_height))
    # Use cv2.addWeighted() to blend the images
    alpha = 0.5
    beta = 1 - alpha
    dst = cv2.addWeighted(roi, alpha, overlay_resized, beta, 0)
    # Replace the ROI in the original image with the blended image
    tansformed_frame[160:1080, 0:1920] = dst

    cv2.namedWindow('Test_Perspectice',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Test_Perspectice', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Test_Perspectice", tansformed_frame)
    cv2.imshow("Test", frame)
    #cv2.imshow("Test_Projector", new_image)
    
    if cv2.waitKey(1) == 27:
        break
