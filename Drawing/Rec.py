import cv2

# Load the image
img = cv2.imread('./pics/new7.jpg')

# Define a mouse callback function
def print_pixel_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the pixel position on the image
        cv2.putText(img, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Image', img)

# Display the image
cv2.imshow('Image', img)

# Set the mouse callback function for the window
cv2.setMouseCallback('Image', print_pixel_position)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
