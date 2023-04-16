import cv2
import numpy as np

# Load the dictionary for the ArUco marker
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

# Create the detector parameters
parameters = cv2.aruco.DetectorParameters_create()

# Create a video capture object for the webcam
cap = cv2.VideoCapture(0)

# Define the IDs of the ArUco markers to detect
marker_ids = [0, 1, 2, 3]

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Detect the markers in the frame
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    # Check if all desired markers were detected
    detected_ids = ids.flatten() if ids is not None else []
    if set(marker_ids).issubset(set(detected_ids)):
        # Extract the corners of the desired markers
        corner_points = []
        for marker_id in marker_ids:
            marker_index = np.where(ids == marker_id)[0][0]
            corner_points.append(corners[marker_index][0])

        # Create a black image with the same dimensions as the frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Create a polygon from the corner points and fill it with white color on the mask
        polygon = np.array(corner_points, np.int32)
        polygon = polygon.reshape((-1,1,2))
        cv2.fillPoly(mask, [polygon], (255, 255, 255))

        # Apply the mask to the frame
        frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

        # Fill the area enclosed by the marker corners with a red color
        red = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(red, [polygon], (0, 0, 255))

        # Add the red color to the masked frame
        result = cv2.add(frame_masked, red)

        # Display the resulting frame
        cv2.imshow('result', result)

    else:
        # Display the original frame if not all markers are detected
        cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
