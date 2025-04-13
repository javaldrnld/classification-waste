import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

# ========================== Setup Logging ==========================
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "camera_inference_final.log")

logging.basicConfig(
    filename=log_file,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

# ========================== TensorFlow Lite Setup ==========================
def setup_tensor(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        logging.info(f"Model loaded: {input_details['name']}")
        return interpreter, input_details, output_details
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None, None

def preprocess_frame(frame, input_det):
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_det["shape"][1], input_det["shape"][2]))
        if input_det["dtype"] == np.float32:
            image = image.astype(np.float32) / 127.5 - 1.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logging.error(f"Error preprocessing frame: {e}")
        return None

# ========================== Parameters and Setup ==========================
model_path = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/models_train/old/mobilenet_v2_standard.tflite"
interpreter, input_details, output_details = setup_tensor(model_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Cannot open camera")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ROI parameters
x1, y1, x2, y2 = 230, 40, 370, 180
roi_area_threshold = 500
time_threshold = 1.0  # seconds
frames_threshold = int(time_threshold * cap.get(cv2.CAP_PROP_FPS))

frame_counter = 0
detection_triggered = False
camera_timeout = 50.0
start_time = time.time()

# ========================== Main Loop ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_with_roi = frame.copy()
    fgmask = fgbg.apply(frame)

    # Extract ROI mask
    roi_mask = fgmask[y1:y2, x1:x2]
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area_roi = sum(cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 300)
    cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if total_area_roi > roi_area_threshold:
        frame_counter += 1
        cv2.putText(frame_with_roi, f"Detecting: {frame_counter}/{frames_threshold}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if frame_counter >= frames_threshold and not detection_triggered:
            detection_triggered = True
            print("Object detected! Running inference...")

            # Run model inference
            input_st = time.perf_counter()
            input_data = preprocess_frame(frame, input_details)
            input_et = time.perf_counter()

            if input_data is not None:
                interpreter.set_tensor(input_details["index"], input_data)

                invoke_st = time.perf_counter()
                interpreter.invoke()
                invoke_et = time.perf_counter()

                output_data = interpreter.get_tensor(output_details["index"])
                score = output_data[0][0]
                class_label = "unacceptable" if score <= 0.50 else "pet_bottle"

                # Display inference result
                text = f"Prediction: {class_label}"
                cv2.putText(frame_with_roi, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # Logging
                logging.info(f"Prediction: {score:.4f} -> {class_label}")
                logging.info(f"Inference time: {(invoke_et - invoke_st) * 1000:.3f} ms")
                logging.info(f"Total time: {(invoke_et - input_st) * 1000:.3f} ms")

    else:
        frame_counter = 0
        cv2.putText(frame_with_roi, "Waiting for object...", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    if not detection_triggered and (current_time - start_time > camera_timeout):
        cv2.putText(frame_with_roi, "Prediction: Wala", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("ROI Mask", roi_mask)
    cv2.imshow("Frame", frame_with_roi)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or detection_triggered:
        break

# ========================== Cleanup ==========================
cap.release()
cv2.destroyAllWindows()
