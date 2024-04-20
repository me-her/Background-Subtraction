import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Initialize video capture with the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Read a frame to get video properties
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read frame")

# Determine a smaller size for faster processing
height, width = frame.shape[:2]
small_size = (int(width * 0.2), int(height * 0.2))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot capture frame")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, small_size)

    # Convert frame to tensor
    tensor_frame = torch.tensor(small_frame).permute(2, 0, 1).float() / 255.0
    tensor_frame = tensor_frame.unsqueeze(0)

    with torch.no_grad():
        prediction = model(tensor_frame)

    # Get the first (and only) batch
    pred = prediction[0]

    # Initialize mask of the same size as the resized frame
    mask = np.zeros(small_size[::-1], dtype="uint8")

    # Process each detected human
    for i, (mask_pred, label) in enumerate(zip(pred["masks"], pred["labels"])):
        if (
            pred["scores"][i] > 0.5 and label.item() == 1
        ):  # Confidence and class filter for humans
            mask_pred = mask_pred[0].mul(255).byte().cpu().numpy()
            mask_pred = cv2.resize(mask_pred, small_size)
            mask = np.maximum(mask, mask_pred)  # Combine masks

    # Resize mask back to original frame size
    mask = cv2.resize(mask, (width, height))

    # Extract foreground
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the resulting frame and foreground
    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground_using_MaskRCNN", foreground)

    # Exit on ESC press
    key = cv2.waitKey(10)
    if key == 27:  # ESC key
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
