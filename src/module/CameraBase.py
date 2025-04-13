import logging

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="./logs/camera_pet_inferencev1.log",
    filemode="a"
)


class CameraBase:
    def __init__(self, camera_id=0):
        """
        Base class for all camera types
        Handles camera setup and basic frame operations
        """
        # Add new camera 
        self.camera_id = camera_id
        self.camera = None
        self.camera_ready = False

    def init_camera(self):
        """
        Initialize the camera. Must be called before using the camera
        """
        if self.camera_ready:
            return
        
        logging.info(f"Initializing camera #{self.camera_id}")
        self.camera = cv2.VideoCapture(self.camera_id)

        if not self.camera.isOpened():
            logging.error(f"Error opening camera {self.camera_id}")
            raise RuntimeError(f"Error opening camera {self.camera_id}")

        # Grab one frame
        ret, frame = self.camera.read()
        if not ret or frame is None:
            logging.error("Capturing initial frame.")
            self.camera.release()
            raise RuntimeError("Error capturing initial frame")
        
        self.camera_ready = True
        logging.info("Camera initialized successfully.")

    
    def release_camera(self):
        """Release thee camera"""
        if self.camera_ready:
            self.camera.release()
            self.camera_ready = False
            logging.info("Camera released")