# Load and preprocess image
import logging
import os

import cv2
import numpy as np
import tensorflow as tf

log_dir = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "image_inference.log")
logging.basicConfig(
    filename=log_file,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

def setup_tensor(model_path):
    """ IMPORTANT DETAILS
        - input_details
            - index 0 -> accepting input
            - shape -> [1, 224, 224, 3] 
                - 1: Batch Size; 224 x 224: Height x Width; 3: Channel (RGB)
            - dtype -> float32: Normalize if intX -> standard
        
        - output_details
            - index 183 -> Position of result -> get after running inference but still not sure if 183 or
            - shape -> [1, 1]: 1 Batch Size; 1 Value output
            - dtype -> Same as input_details
    """
    # Load the model
    # interpreter = tf.lite.Interpreter("/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/models_train/mobilenetv2/mobilenetv2_best_test_3_standard.tflite")
    try:
        interpreter = tf.lite.Interpreter(model_path)
        # Allocate memory for IO -> Always needed before execution
        interpreter.allocate_tensors()

        # Input should be [, 224, 224, 3] -> Image should be normalize first 
        input_details = interpreter.get_input_details()[0]
        # Result at index: 183
        output_details = interpreter.get_output_details()[0]

        logging.info(f"Model loaded: {input_details['name']}")
        logging.info(f"INPUT DETAILS: Index - {input_details['index']} | Shape - {input_details['shape']} | dtype - {input_details['dtype']}")
        logging.info(f"OUTPUT DETAILS: Index - {output_details['index']} | Shape - {output_details['shape']} | dtype - {output_details['dtype']}")

        return interpreter, input_details, output_details
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None, None

def preprocess_image(image_path, input_det):
    try:
    # Load image using OpenCV
        image = cv2.imread(image_path)
        # print(image.dtype)
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image.dtype)

        # Resize image to match the model -> 224, 224,
        image = cv2.resize(image, (input_det["shape"][1], input_det["shape"][2]))

        # Normalize incase the model normalize
        if input_det["dtype"] == np.float32:
            image = image.astype(np.float32) / 127.5 - 1.0 #255.0 # 255 since 0-255 color
            logging.info(f"Image {image_path} normalized to range: {image.min():.2f} to {image.max():.2f}")
        
        # Expand dimension to match
        image = np.expand_dims(image, axis=0)
        logging.info(f"Preprocessed image shape: {image.shape}")
        return image
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None
    
# Create an inference to run on the list
def run_inference(img_dir, model_path):
    interpreter, input_details, output_details = setup_tensor(model_path)

    if interpreter is None:
        logging.error("Failed to load model, exiting")
        return
    try:
        for image_name in [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]:
            full_img_path = os.path.join(img_dir, image_name)
            # Preprocess image
            input_data = preprocess_image(full_img_path, input_details)
            logging.info(f"Preprocessing {full_img_path}")

            if input_data is None:
                continue

            # Set input tensor -> Use the input index to get the index tensor
            interpreter.set_tensor(input_details["index"], input_data)

            # Run inference
            interpreter.invoke()

            # Get the output index using the output details
            # print(model_path["index"])
            output_data= interpreter.get_tensor(output_details["index"])
            # print(output_data[0][0])
            class_label = "unacceptable" if output_data[0][0] <=0.50 else "pet_bottle"
            print(f"Prediction for {image_name}: {output_data[0][0]:.4f} -> {class_label}")
            logging.info(f"Prediction for {image_name}: {output_data[0][0]:.4f} -> {class_label}")

    except Exception as e:
        logging.error(f"Error loading model: {e}")
    
if __name__ == "__main__":
    img_directory = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/data/new_test"
    model_directory = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/models_train/mobilenetv2/mobilenetv2_best_test_3_standard.tflite"
    run_inference(img_directory, model_directory)