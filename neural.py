from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import cv2
import numpy as np

# Initialize video capture with the default webcam.
cap = cv2.VideoCapture(0)

# Load the model and feature extractor.
model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)


def segment_frame(frame):
    # Convert the captured frame to PIL Image format.
    pil_img = Image.fromarray(frame)

    # Prepare the frame using the feature extractor.
    inputs = feature_extractor(images=pil_img, return_tensors="pt")

    # Forward pass, get the logits and upsample them.
    outputs = model(**inputs)
    logits = outputs.logits  # Shape: (batch_size, num_labels, height, width)
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False
    )

    # Convert logits to probabilities and get the predicted segmentation.
    probs = upsampled_logits.softmax(dim=1)[0]  # Take the first item in the batch.
    predicted_segmentation = probs.argmax(dim=0).cpu().numpy()

    # Here you can further process `predicted_segmentation` to isolate the foreground.
    # This will depend on the model and your specific needs.

    return predicted_segmentation


while True:
    ret, frame = cap.read()
    if not ret:
        break  # If no frame is captured or end of video is reached, break out of the loop

    # Process the frame through the neural network.
    segmentation = segment_frame(frame)

    # You'll need to adapt the following to display your results depending on the model's output.
    cv2.imshow("Segmentation", segmentation)

    key = cv2.waitKey(30)
    if key == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
