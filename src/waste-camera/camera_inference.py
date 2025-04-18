import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

log_dir = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "camera_image_inference.log")
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

    try:
        interpreter = tf.lite.Interpreter(model_path)
        # Always allocate memory for IO
        interpreter.allocate_tensors()

        # Input should be [, 224, 224, 3] -> Normalize first [-1, 1]
        input_details = interpreter.get_input_details()[0]
        # Result index last time checked is 183
        output_details = interpreter.get_output_details()[0]

        logging.info(f"Model loaded: {input_details['name']}")
        logging.info(f"INPUT DETAILS: Index - {input_details['index']} | Shape - {input_details['shape']} | dtype - {input_details['dtype']}")
        logging.info(f"OUTPUT DETAILS: Index - {output_details['index']} | Shape - {output_details['shape']} | dtype - {output_details['dtype']}")

        return interpreter, input_details, output_details
    
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None, None

def preprocess_frame(frame, input_det):
    try:
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize image
        image = cv2.resize(image, (input_det["shape"][1], input_det["shape"][2]))
        # Normalize to [-1, 1]
        if input_det["dtype"] == np.float32:
            image = image.astype(np.float32) / 255.0
            logging.info(f"Frame normalized to range: {image.min():.2f} to {image.max():.2f}")

        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logging.error(f"Error preprocessing frame: {e}")
        return None

def run_camera_inference(model_path):
    interpreter, input_details, output_details = setup_tensor(model_path)

    if interpreter is None:
        logging.error("Failed to load model, exiting ...")

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return
    
    try:
        while True:
            # read() return false if there's no frame
            # ret -> Check whether the webcam is availble or not
            ret, frame = cap.read()
            logging.info(f"Frame shape: {frame.shape}")

            if not ret:
                logging.error("Failed to capture frame")
                break
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) &0xFF
            
            if key == ord("c"):
                # Implement input here 
                input_st = time.perf_counter()

                input_data = preprocess_frame(frame, input_details)

                input_et = time.perf_counter()
                logging.info("Preprocessing Input data")
                input_total_t = (input_et - input_st) * 1000
                logging.info(f"Preprocessing time: {input_total_t:.3f} ms")
                print(f"Preprocessing time: {input_total_t:.3f} ms")

                if input_data is None:
                    continue
                
                # Set input tensor -> It will use for inputting capture image to tensor
                interpreter.set_tensor(input_details["index"], input_data)

                # Run inference
                invoke_st = time.perf_counter()

                interpreter.invoke()
                invoke_et = time.perf_counter()

                invoke_total_t = (invoke_et - invoke_st) * 1000
                logging.info(f"Inference (invoke) time: {invoke_total_t:.3f} ms")
                print(f"Inference (invoke) time: {invoke_total_t:.3f} ms")

                # Get the output index using output details
                # Get the predicted result
                output_data = interpreter.get_tensor(output_details["index"])
                score = output_data[0][0]

                class_label = "unacceptable" if score <= 0.50 else "pet_bottle"
                total_time = (invoke_et - input_st) * 1000
                logging.info(f"Total Inference Time: {total_time:.3f} ms")
                print(f"Total Inference time: {total_time:.3f} ms")

                print(f"Prediction: {score:.4f} -> {class_label}")
                # headless remove this line below
                copy_frame = frame.copy()
                text = f"Prediction: {class_label}"
                org = (50, 100)
                fontFace = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255,255,255)
                cv2.putText(copy_frame, text, org, fontFace, fontScale, color)
                cv2.imshow("result:", copy_frame)


                logging.info(f"Prediction: {score:.4f} -> {class_label}")
                logging.info(cap.get(cv2.CAP_PROP_FPS))
            
            # Change to unacceptable -> Break
            elif key == ord("q"):
                break

        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"FPS: {1 / total_time}")


            
    except Exception as e:
        logging.error(f"Error during inference: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Camera released and windows closed")
    
model_path = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/models_train/old/mobilenet_v2_standard.tflite"
run_camera_inference(model_path)  