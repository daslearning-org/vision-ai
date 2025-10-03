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
label_file_path = os.path.join(base_path, 'spicesNet_labels_v401a.txtset')


class OnnxSpecies():
    def __init__(self, save_dir=save_path, model_dir=models_dir):
        self.model_flag = False
        self.sess = None
        self.save_dir = save_dir
        self.model_dir = model_dir
        with open(label_file_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def start_species_session(self, model_name="spicesNet_v401a.onnx"):
        model_path = os.path.join(self.model_dir, model_name)
        download_path = os.path.join(self.model_dir, "spicesNet_v401a.onnx")
        if os.path.exists(download_path):
            model_path = download_path
            self.model_flag = True
        elif os.path.exists(model_path):
            self.model_flag = True
        else:
            model_path = download_path
            print(f"The onnx model: {model_path} does not exist! Downloading it now...")
            # try to download the file from github as a backup
            downlaod_url = "https://github.com/daslearning-org/vision-ai/releases/download/vOnnxModels/spicesNet_v401a.onnx"
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
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        return False

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (480, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.expand_dims(img, axis=0)  # (1, 480, 480, 3)
        return img

    def postprocess_logits(self, logits, labels):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0, predicted_class]
        return predicted_class, confidence, labels[predicted_class]

    def run_species(self, image_path, callback=None, caller=None):
        final_result = {"status": False, "message": "Initial load", "caller": caller}
        if self.sess is None:
            final_result['message'] = "Onnx session was not initialized! Check if model has been downloaded."
            return final_result
        # Load and preprocess the image
        image_filename = image_path.split("/")[-1]

        try:
            img = self.preprocess_image(image_path)

            # run the classification
            outputs = self.sess.run(None, {self.input_name: img})
            predicted_class, confidence, predicted_label = self.postprocess_logits(outputs[0], self.labels)
            confidence = confidence*100

            # create the return label
            if int(predicted_class) == 2246:
                label = "The image does not contain any object that falls into the list!"
            else:
                label = f"Species: [b][color=#2574f5]{predicted_label[37:]}[/color][/b] \n"
                label = label + f"Confidence: [b][color=#2574f5]{confidence:.2f}% [/color][/b]"
            final_result["message"] = label
            final_result["status"] = True
        except Exception as e:
            print(f"Classification error: {e}")
            final_result["message"] = f"Classification error: {e}"

        if callback:
            Clock.schedule_once(lambda dt: callback(final_result))
        else:
            return final_result
