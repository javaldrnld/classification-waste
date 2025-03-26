import threading
import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite  # Use if only using TFLite runtime

# from tensorflow.lite.python.interpreter import Interpreter  # Use if using full TensorFlow
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtGui import QImage


class DetectionLogic(QObject):
    new_frame_signal = pyqtSignal(QImage)
    result_signal = pyqtSignal(str, str)

    def __init__(self, model_path="model.tflite"):
        super().__init__()
        
        self.camera_active = False
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.threshold_y = 300
        self.detection_cooldown = False

        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get model input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Timer for frame processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def start_camera(self):
        """Start the camera and frame processing."""
        self.camera_active = True
        self.cap = cv2.VideoCapture(3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.timer.start(30)

    def stop_camera(self):
        """Stop the camera and release resources."""
        self.camera_active = False
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def process_frame(self):
        """Capture frames and process them."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                display_frame = frame.copy()

                # Draw threshold line
                cv2.line(display_frame, (0, self.threshold_y), 
                         (display_frame.shape[1], self.threshold_y), (0, 255, 0), 2)

                # Run detection
                if not self.detection_cooldown:
                    detection_thread = threading.Thread(
                        target=self.run_pet_bottle_detection, args=(frame.copy(),)
                    )
                    detection_thread.daemon = True
                    detection_thread.start()

                # Convert frame to QImage for display
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
                self.new_frame_signal.emit(qt_image)

    def preprocess_frame(self, frame):
        """Resize and normalize frame for model input."""
        input_shape = self.input_details[0]['shape']  # Model input shape
        input_size = (input_shape[1], input_shape[2])  # Example: (224, 224)
        
        img = cv2.resize(frame, input_size)
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def run_pet_bottle_detection(self, frame):
        """Run inference on the frame using the TFLite model."""
        input_tensor = self.preprocess_frame(frame)

        # Set model input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()  # Run inference

        # Get model output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        detection_result = np.argmax(output_data)  # Assuming classification model

        # Define labels (modify according to your model’s output classes)
        labels = ["Not a PET Bottle", "PET Bottle"]
        detected_label = labels[detection_result]

        # Update UI with detection result
        if detected_label == "PET Bottle":
            self.result_signal.emit("PET Bottle Detected", "#4CAF50")  # Green
        else:
            self.result_signal.emit("No PET Bottle", "#F44336")  # Red

        # Cooldown to avoid duplicate detections
        self.detection_cooldown = True
        threading.Thread(target=self.reset_cooldown).start()

    def reset_cooldown(self):
        """Reset the detection cooldown after a delay."""
        time.sleep(2)
        self.detection_cooldown = False

    def cleanup(self):
        """Cleanup resources before exiting."""
        self.stop_camera()
