import cv2
import numpy as np
# Load the two images
image1 = cv2.imread('./pics/new.jpg')
image2 = cv2.imread('./pics/new1.jpg')

# Get the size of the first image
height1, width1 = image1.shape[:2]

# Get the size of the second image
height2, width2 = image2.shape[:2]

# Create a new image with a height equal to the sum of the heights of the two images, and a width equal to the maximum width of the two images
new_image = np.zeros((height1 + height2, max(width1, width2), 3), np.uint8)

# Paste the first image onto the new image, starting at the top left corner
new_image[:height1, :width1] = image1

# Paste the second image onto the new image, starting below the first image
new_image[height1:height1+height2, :width2] = image2
new_image = cv2.resize(new_image, (720, 960))
# Display the new image
cv2.imshow("Merged Image", new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()