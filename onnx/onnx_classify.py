import cv2
import numpy as np
from onnxruntime import InferenceSession
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
synset_path = os.path.join(base_path, 'synset_words.txtset')

# Load the model
model_path_local = 'resnet18-v1-7.onnx'  # Ensure this path is correct

class OnnxClassify():
    def __init__(self, save_dir=save_path, model_dir=model_path_local):
        self.model_flag = False
        self.sess = None
        self.save_dir = save_dir
        self.model_dir = model_dir

    def start_detect_session(self, model_name="resnet18-v1-7.onnx"):
        model_path = os.path.join(self.model_dir, model_name)
        download_path = os.path.join(self.model_dir, "resnet18-v1-7.onnx")
        if os.path.exists(download_path):
            model_path = download_path
            self.model_flag = True
        elif os.path.exists(model_path):
            self.model_flag = True
        else:
            model_path = download_path
            print(f"The onnx model: {model_path} does not exist! Downloading it now...")
            # try to download the file from github as a backup
            downlaod_url = f"https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx"
            import requests
            try:
                with requests.get(downlaod_url, stream=True) as req:
                    req.raise_for_status()
                    with open(download_path, 'wb') as f:
                        for chunk in req.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Onnx file downloaded successfully to: {download_path} 🎉")
                self.model_flag = True
            except requests.exceptions.RequestException as e:
                print(f"Error downloading the onnx file: {e} 😞")

        if self.model_flag:
            try:
                self.sess = InferenceSession(model_path)
                # Get input and output names
                self.input_name = self.sess.get_inputs()[0].name
                self.output_name = self.sess.get_outputs()[0].name
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        return False

    def softmax(x):
        # Compute softmax probabilities
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def run_classify(self, image_path, callback=None, caller=None):
        final_result = {"status": False, "message": "Initial load", "caller": caller}
        if self.sess is None:
            final_result['message'] = "Onnx session was not initialized! Check if model has been downloaded."
            return final_result
        # Load and preprocess the image
        image_filename = image_path.split("/")[-1]

        try:
            img = cv2.imread(image_path)
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize to 224x224 (standard for ImageNet models like MobileNetV2)
            img = cv2.resize(img, (224, 224))
            # Convert to float32 and normalize
            img = img.astype(np.float32)
            # Standard ImageNet normalization, ensure mean and std are float32
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
            img = (img - mean) / std
            # Transpose to [C, H, W] and add batch dimension [1, C, H, W]
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            # Ensure the final input is float32
            img = img.astype(np.float32)

            # use the labels from synset
            with open(synset_path, 'r') as f:
                labels = [line.split(' ', 1)[1].strip() for line in f.readlines()]

            # run the classification
            outputs = self.sess.run([self.output_name], {self.input_name: img})
            probabilities = self.softmax(outputs[0])
            top5_indices = np.argsort(probabilities[0])[::-1][:5]
            top5_probs = probabilities[0][top5_indices]
            # create the return label
            label = "Top-5 predictions: \n"
            for i, idx in enumerate(top5_indices):
                label = label + f"{i+1}. {labels[idx]}: {top5_probs[i]:.4f} \n"
            final_result["message"] = label
            final_result["status"] = True
        except Exception as e:
            print(f"Classification error: {e}")
            final_result["message"] = f"Classification error: {e}"

        if callback:
            Clock.schedule_once(lambda dt: callback(final_result))
        else:
            return final_result
