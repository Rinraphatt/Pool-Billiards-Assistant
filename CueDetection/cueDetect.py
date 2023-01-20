from __future__ import print_function
import cv2
import argparse

# cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

cv2.namedWindow("Python Webcam Screenshot App")

img_counter = 0

# Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

img = cv2.imread("./pics/pool_table_ball.jpg")

while True:
    # ret,frame = cam.read()

    # if not ret: 
    #     print("failed to grap frame")
    #     break

    # Applying the Canny Edge filter
    # edge1 = cv2.Canny(frame, t_lower, t_upper)
    # edge2 = cv2.Canny(frame, 250, 350)

    edge1 = cv2.Canny(img, t_lower, t_upper)
    edge2 = cv2.Canny(img, 250, 350)

    cv2.imshow('original', img)
    cv2.imshow('edge1', edge1)
    cv2.imshow('edge2', edge2)

    
    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing the app")
        break
    # elif k%256 == 32:
        # img_name = "LangSad_pic_{}.png".format(img_counter)
        # cv2.imwrite(img_name,frame)
        # print("screenshot taken")
        # img_counter += 1

# cam.release()

# cam.destroyAllWindows()

# img = cv2.imread("pool_table.jpeg")  # Read image
  
# # Setting parameter values
# t_lower = 50  # Lower Threshold
# t_upper = 150  # Upper threshold
  
# # Applying the Canny Edge filter
# edge = cv2.Canny(img, t_lower, t_upper)
  
# cv2.imshow('original', img)
# cv2.imshow('edge', edge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()