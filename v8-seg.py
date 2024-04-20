import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

# from ultralytics.yolo.utils.ops import scale_image

# Load the YOLOv8n-seg model
seg_model = YOLO("yolov8n-seg.pt")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform segmentation on the frame
    res = seg_model(frame, conf=0.5, show=False)

    # iterate detection results
    for r in res:
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem

        # iterate each object contour
        for ci, c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]

            b_mask = np.zeros(img.shape[:2], np.uint8)

            # Create contour mask
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

            # OPTION-1: Isolate object with black background
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)
            # Display the segmented image
            cv2.imshow("Segmentation", isolated)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
