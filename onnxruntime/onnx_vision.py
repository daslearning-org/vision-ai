import cv2
import numpy as np
import onnxruntime as rt

# Load the model
model_path = 'ssd_mobilenet_v1.onnx'  # Ensure this path is correct
try:
    sess = rt.InferenceSession(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Get input and output names
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

# Print input and output details for debugging
print(f"Input name: {input_name}, shape: {sess.get_inputs()[0].shape}, type: {sess.get_inputs()[0].type}")
print(f"Output names: {output_names}")
for i, output in enumerate(sess.get_outputs()):
    print(f"Output {i}: name={output.name}, shape={output.shape}")

# COCO class labels
coco_labels = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}


# Load and preprocess the image
image_path = 'your_image.jpg'  # Replace with your image path
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image at {image_path}")
    exit(1)
original_height, original_width = img.shape[:2]

# Resize to 300x300 for model input, keep as RGB uint8
img_resized = cv2.resize(img, (300, 300))
img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Add batch dimension: shape (1, 300, 300, 3), keep uint8
img_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
print(f"Input data shape: {img_data.shape}, type: {img_data.dtype}")

# Run inference
try:
    results = sess.run(output_names, {input_name: img_data})
except Exception as e:
    print(f"Inference error: {e}")
    exit(1)

# Inspect output shapes
for i, (name, output) in enumerate(zip(output_names, results)):
    print(f"Output {name} shape: {np.array(output).shape}, type: {np.array(output).dtype}")

# Parse outputs
detection_boxes = results[0][0]  # Shape (100, 4): [y1, x1, y2, x2] normalized [0,1]
detection_classes = results[1][0]  # Shape (100,): class indices
detection_scores = results[2][0]  # Shape (100,): confidence scores
num_detections = results[3]  # Shape (1,): number of detections

# Extract num_detections as scalar
if num_detections.size == 1:
    num_detections = int(num_detections.item())
else:
    print(f"Error: Unexpected num_detections shape {num_detections.shape}")
    exit(1)

# Prepare original image for drawing (convert to RGB then back to BGR for OpenCV)
output_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert original image to RGB
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR

# Filter detections by score threshold and draw boxes on original image
threshold = 0.5
for i in range(min(num_detections, len(detection_scores))):
    score = detection_scores[i]
    if score > threshold:
        class_id = int(detection_classes[i])
        label = coco_labels.get(class_id, 'unknown')
        box = detection_boxes[i]

        # Scale boxes to original image size
        y1 = int(box[0] * original_height)
        x1 = int(box[1] * original_width)
        y2 = int(box[2] * original_height)
        x2 = int(box[3] * original_width)

        # Draw rectangle and label
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_img, f"{label}: {score:.2f}", (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save or display
cv2.imwrite('output_image.jpg', output_img)
cv2.imshow('Detections', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()