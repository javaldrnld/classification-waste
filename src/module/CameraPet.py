import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

from src.module.CameraBase import CameraBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="camera_pet_inference.log",
    filemode="a"
)

class CameraPet(CameraBase):
    def __init__(self, camera_id=0, model_path=None, use_tflite=True):
        """
        Initialize the Camera for PET bottle classisifcation.
        """
        load_dotenv()
        super().__init__(camera_id)

        # Set model path
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.use_tflite = use_tflite
        if not self.model_path:
            logging.error("Model path not provided")
            raise ValueError(("Model path not provided"))

        # Initialize parameters for PET Bottle
        self.fgbg = None
        self.detection_triggered = None
        self.frame_counter = 0
        self.prediction_result = None

        # Configure ROI and thresholds
        self.roi_config = {
            "x1": 230, "y1": 40,
            "x2": 370, "y2": 180,
            "area_threshold": 500,
            "time_threshold": 1.0 # In seconds
        }
        
        # Setup Morphological kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        # Setup TFLite model
        self._setup_model()

    # Alam ko gumagana na private class nito kapag double underscore
    def _setup_model(self):
        """
        Initialize the TF for Classification or HF
        """
        if self.use_tflite:
            try:
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()

            # Get input and output details
                self.input_details = self.interpreter.get_input_details()[0]
                self.output_details = self.interpreter.get_output_details()[0]
            
                logging.info(f"Model loaded: {self.input_details['name']}")
                logging.info(f"INPUT DETAILS: Shape - {self.input_details['shape']} | dtype - {self.input_details['dtype']}")
                logging.info(f"OUTPUT DETAILS: Shape - {self.output_details['shape']} | dtype - {self.output_details['dtype']}")                # Get input and output details
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                raise
        else:
            try:
                self.model = load_model(self.model_path)
                logging.info("Loaded Keras .h5 model")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                raise
            
    def _preprocess_frame(self, frame):
        """
        Preprocess a frame for model inference
        """
        try:
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            if self.use_tflite:
                # Resize image to match model input for TFLite
                image = cv2.resize(image, (self.input_details["shape"][1], self.input_details["shape"][2]))
            
                # Normalize to [-1, 1] for MobileNet/EfficientNet
                if self.input_details["dtype"] == np.float32:
                    # image = image.astype(np.float32) / 127.5 - 1.0
                    # Update to [0, 1] -> According to YOLO
                    image = image.astype(np.float32) / 255.0
            else:
                # For Keras .h5 model - resize to expected input shape (224, 224) by default
                image = cv2.resize(image, (224, 224))
            
                # Normalize to [-1, 1] for .h5 model
                image = image.astype(np.float32) / 127.5 - 1.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            return image
        except Exception as e:
            logging.error(f"Error preprocessing frame: {e}")
            return None

    def _run_inference(self, frame):
        """
        Run model inference on a single frame
        """
        try:
            # Preprocecss frame
            input_data = self._preprocess_frame(frame)
            if input_data is None:
                return None, None, None
            
            # Start time measurement
            start_time = time.perf_counter()

            # Set input tensor
            if self.use_tflite:
                self.interpreter.set_tensor(self.input_details["index"], input_data)
            
                # Run inference
                self.interpreter.invoke()
            
                # Get output and calculate class
                # Change since YOLO produce multiclass label
                output_data = self.interpreter.get_tensor(self.output_details["index"])
                output = output_data[0]
                pred_class = int(np.argmax(output_data))
                confidence = output[pred_class]
                # score = output_data[0][0]

            else:
                output_data = self.model.predict(input_data)
                score = float(output_data[0][0])
            
            # Determine class based on score threshold
            # class_label = "unacceptable" if score <= 0.50 else "pet_bottle"
            label_map = {0: "unacceptable", 1: "pet_bottle"}
            class_label = label_map[pred_class]
            
            # Calculate total time
            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            
            logging.info(f"Inference time: {inference_time:.3f} ms")
            logging.info(f"Prediction: {confidence:.4f} -> {class_label}")
            
            return class_label, confidence, inference_time
            
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return None, None, None

    def _setup_detection(self):
        """Initialize the background subtractor"""

        # Create background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        # Calculate frame threshold based on camera fps
        if self.camera_ready:
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            self.frames_threshold = int(self.roi_config["time_threshold"] * fps)
        else:
            # Defeault to 30 fps if camera not initialzied
            self.frames_threshold = int(self.roi_config["time_threshold"] * 30)

    def _process_roi(self, frame):
        """Process the region of interest to detect object presence"""
        fgmask = self.fgbg.apply(frame)
        # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        # fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
        fgmask = cv2.dilate(fgmask, self.kernel, iterations=1)

        # Extrract the ROI from the mask
        x1, y1 = self.roi_config["x1"], self.roi_config["y1"]
        x2, y2 = self.roi_config["x2"], self.roi_config["y2"]
        roi_mask = fgmask[y1:y2, x1:x2]

        # Find contours in the ROI
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate total area of significant contours
        total_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:  # Filter small noise
                total_area += area
                
        return roi_mask, total_area 

    # Temporary value: Dispaly -> Show ROI frame
    def infer(self, timeout=10.0, display=True):
        """Main method to detect and classifyt PET bottled"""

        logging.info("Starting PET bottle detection and classification")
         
        # Initialize camera
        if not self.camera_ready:
            self.init_camera()

        # Setup detection paramters
        self._setup_detection()
        
        # Reset detection state
        self.detection_triggered = False
        self.frame_counter = 0
        # For timeout
        start_time = time.time()

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    logging.error("Failed to capture frame")
                    break

                # Check for timeout
                current_time = time.time()
                if not self.detection_triggered and (current_time - start_time > timeout):
                    self.prediction_result = {
                        "class": 'unacceptable',
                        "score": 0.0,
                        "inference_time": 0.0,
                        "reason": "no object",
                    }
                    self.detection_triggered = True
                    logging.info("Detection timeout reached or bottle")
                    break
            
                # Create a copy for visualziation
                frame_with_roi = frame.copy()

                # PRocess the region of intereset
                roi_mask, total_area = self._process_roi(frame)
                # Display frame with ROI
                if display:
                    x1, y1 = self.roi_config['x1'], self.roi_config['y1']
                    x2, y2 = self.roi_config['x2'], self.roi_config['y2']
                    cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Object persistence check
                if total_area > self.roi_config['area_threshold']:
                    self.frame_counter += 1
                    
                    # Show detection progress
                    if display:
                        cv2.putText(frame_with_roi, f"Detecting: {self.frame_counter}/{self.frames_threshold}", 
                                   (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Check if detection threshold reached
                    if self.frame_counter >= self.frames_threshold and not self.detection_triggered:
                        # Run inference
                        class_label, score, inference_time = self._run_inference(frame)
                        
                        if class_label is not None:
                            self.prediction_result = {
                                "class": class_label,
                                "score": float(score),
                                "inference_time": float(inference_time),
                                "reason": "detected"
                            }
                            self.detection_triggered = True
                            
                            # Display prediction
                            if display:
                                cv2.putText(frame_with_roi, f"Prediction: {class_label} ({score:.2f})", 
                                           (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                            
                            break
                else:
                    # Reset counter if object disappears
                    self.frame_counter = 0
                    if display:
                        cv2.putText(frame_with_roi, "Waiting for object...", (30, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                
                # Display frames if requested
                if display:
                    cv2.imshow("ROI Mask", roi_mask)
                    cv2.imshow("Frame", frame_with_roi)
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    # Force manual classification
                    class_label, score, inference_time = self._run_inference(frame)
                    
                    if class_label is not None:
                        self.prediction_result = {
                            "class": class_label,
                            "score": float(score),
                            "inference_time": float(inference_time),
                            "reason": "manual"
                        }
                        
                        if display:
                            result_frame = frame.copy()
                            cv2.putText(result_frame, f"Manual: {class_label} ({score:.2f})", 
                                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.imshow("Manual Result", result_frame)
                        
                        break
            
            return self.prediction_result
            
        except Exception as e:
            logging.error(f"Error during operation: {e}")
            return {"class": "error", "reason": str(e)}
        finally:
            # Clean up windows but don't release camera
            if display:
                cv2.destroyAllWindows()
    
    def get_last_prediction(self):
        """
        Get the last prediction result.
        
        Returns:
            dict: Last prediction result or None if no prediction made
        """
        return self.prediction_result

                
                
