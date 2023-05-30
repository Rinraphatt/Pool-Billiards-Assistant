import cv2

# Create a VideoCapture object for the default webcam
cap = cv2.VideoCapture(0)

# Check if the webcam was successfully opened
if not cap.isOpened():
    print("Unable to open webcam")
    exit()

# Loop over frames from the webcam
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error capturing frame")
        break

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Check for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
