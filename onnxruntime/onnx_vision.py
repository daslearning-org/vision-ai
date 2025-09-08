import cv2
import numpy as np
import onnxruntime as rt
from kivy.clock import Clock

import os, sys

# Determine the base path for your application's resources
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_path, "model_files")
save_path = os.path.join(base_path, 'outputs')

# Load the model
model_path_local = 'ssd_mobilenet_v1.onnx'  # Ensure this path is correct

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


class OnnxDetect():
    def __init__(self, save_dir=save_path, model_dir=model_path_local, model_name="ssd_mobilenet_v1.onnx"):
        self.session_flag = False
        self.save_dir = save_dir
        self.model_dir = model_dir
        model_path = os.path.join(self.model_dir, model_name)
        try:
            self.sess = rt.InferenceSession(model_path)
            # Get input and output names
            self.input_name = self.sess.get_inputs()[0].name
            self.output_names = [o.name for o in self.sess.get_outputs()]
        except Exception as e:
            print(f"Error loading model: {e}")
            self.session_flag = False
            self.sess = None

    def run_detect(self, image_path, callback=None):
        final_result = {"status": False, "message": "Initial load"}
        # Load and preprocess the image
        image_filename = image_path.split("/")[-1]
        op_img_path = os.path.join(self.save_dir, f"op-{image_filename}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            final_result['message'] = f"Error: Could not load image at {image_path}"
            return final_result
        original_height, original_width = img.shape[:2]

        # Resize to 300x300 for model input, keep as RGB uint8
        img_resized = cv2.resize(img, (300, 300))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Add batch dimension: shape (1, 300, 300, 3), keep uint8
        img_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
        print(f"Input data shape: {img_data.shape}, type: {img_data.dtype}")

        # Run inference
        try:
            results = self.sess.run(self.output_names, {self.input_name: img_data})
        except Exception as e:
            print(f"Inference error: {e}")
            final_result['message'] = f"Inference error: {e}"
            return final_result

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
            final_result['message'] = f"Error: Unexpected num_detections shape {num_detections.shape}"
            return final_result

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
        cv2.imwrite(op_img_path, output_img)
        final_result['status'] = True
        final_result['message'] = op_img_path

        if callback:
            Clock.schedule_once(lambda dt: callback(final_result))
        else:
            return final_result
