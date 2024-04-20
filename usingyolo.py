import cv2
import numpy as np
import torch
from yolov5 import YOLOv5

# Load the YOLOv5 model
# Assuming "yolov5s.pt" is in your working directory

model = YOLOv5("yolov5s.pt")


# Initialize video capture with the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    detections = model.predict(frame)
    boxes = detections.xyxy[0]  # Bounding boxes

    # Initialize mask of the same size as frame
    mask = np.zeros(frame.shape[:2], dtype="uint8")

    # Draw rectangles around detected objects and update mask
    for box in boxes:
        x1, y1, x2, y2, conf, cls = map(int, box[:6])
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 0, 0), 2
        )  # Draw rectangle on frame
        cv2.rectangle(
            mask, (x1, y1), (x2, y2), (255), -1
        )  # Update mask to include detected object

    # Extract foreground
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the resulting frame and foreground
    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground_using_YOLOv5", foreground)

    # Exit on ESC press
    key = cv2.waitKey(10)
    if key == 27:  # 27 is the ESC key
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
