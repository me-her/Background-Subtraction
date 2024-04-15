import cv2
import numpy as np

# Initialize video capture with the default webcam.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize the background subtractor.
subtractor = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=0, detectShadows=True
)

while True:
    ret, frame = cap.read()
    print(type(frame))
    if not ret:
        print("Cannot capture frame")
        break  # If no frame is captured or end of video is reached, break out of the loop

    # Apply the background subtractor to get the mask.
    mask = subtractor.apply(frame)

    # Optional: Remove the noise and fill the holes in the mask for a cleaner output.
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=2)

    # Apply the mask to the frame using bitwise_and to keep only the foreground.
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the original frame, the mask, and the foreground.
    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)
    cv2.imshow("Foreground", foreground)

    key = cv2.waitKey(10)
    break
    if key == 9:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
