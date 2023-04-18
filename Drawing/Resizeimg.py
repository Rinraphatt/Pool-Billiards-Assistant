import cv2

# Create a video capture object for the webcam
width = 1920
height = 1080
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# Get the webcam frame dimensions
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))



while True:
    # Read a frame from the webcam
    ret, frame = video.read()

    if ret:
        # Determine the scale factor
        scale_factor = min(1920/frame_width, 1080/frame_height)

        # Resize the frame
        resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # Create a new blank image with a resolution of 1920x1080
        new_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Calculate the position of the resized frame in the new frame
        x_offset = int((1920 - resized_frame.shape[1])/2)
        y_offset = int((1080 - resized_frame.shape[0])/2)

        # Paste the resized frame into the new frame
        new_frame[y_offset:y_offset+resized_frame.shape[0], x_offset:x_offset+resized_frame.shape[1]] = resized_frame


        # Display the new frame
        cv2.imshow('New Frame', new_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release the video capture and writer objects, and close all windows
video.release()
writer.release()
cv2.destroyAllWindows()
