import numpy as np
import cv2

vidcap = cv2.VideoCapture(0)
width = 1920
height = 880
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
black = cv2.imread('./pics/Black.jpg')
mac = cv2.imread('./iMac - 18.png')

while True:
    succuess, img = vidcap.read()
    frame = img
    mac = cv2.resize(mac, (1920, 880))
    tl = (212 ,180)
    bl = (159 ,923)
    tr = (1696 ,178)
    br = (1750 ,925)
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

    # Get the size of the first image
    height1, width1 = mac.shape[:2]

    # Get the size of the second image
    height2, width2 = black.shape[:2]
    # Create a new image with a height equal to the sum of the heights of the two images, and a width equal to the maximum width of the two images
    new_image = np.zeros((1080, 1920, 3), np.uint8)

    # Paste the first image onto the new image, starting at the top left corner
    new_image[:height2, :width2] = black

    # Paste the second image onto the new image, starting below the first image
    new_image[height2:height2+height1, :width2] = mac

     # Paste the second image onto the new image, starting below the first image
    new_image[height2+height1:height2+height1+height2, :width2] = black
    print(new_image.shape[:2])
    cv2.namedWindow('Test_Projector',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Test_Projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Test_Perspectice", tansformed_frame)
    cv2.imshow("Test", frame)
    cv2.imshow("Test_Projector", new_image)
    
    if cv2.waitKey(1) == 27:
        break
