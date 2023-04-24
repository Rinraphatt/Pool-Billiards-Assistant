import cv2

# Create a video capture object
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# Create a window with trackbars to adjust threshold values
cv2.namedWindow('Controls')
cv2.createTrackbar('Threshold', 'Controls', 50, 255, lambda x: None)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Subtract the grayscale frame from a reference frame
    ref_gray = cv2.imread('./pics/Noball_W.jpg', cv2.IMREAD_GRAYSCALE)
    diff = cv2.absdiff(ref_gray, gray)

    # Apply a threshold to create a binary mask
    threshold = cv2.getTrackbarPos('Threshold', 'Controls')
    threshold1 = cv2.getTrackbarPos('Threshold', 'Controls')
    thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(thresh, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    print(mask)
    # Find the contours of the ball blobs   
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours based on their area, circularity, and aspect ratio
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * 3.14159 * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if area > 100 and circularity > 0.6 and aspect_ratio < 1.2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the original frame with detected balls
    cv2.imshow('Detected balls', frame)
    cv2.imshow('Detected balls1', mask)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy the windows
cap.release()
cv2.destroyAllWindows()
