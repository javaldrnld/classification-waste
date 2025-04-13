import logging
import os
import time

import cv2 as cv
import numpy as np
import tensorflow as tf

# === Setup Logging ===
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "camera_inference.log")
logging.basicConfig(
    filename=log_file,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

# === TensorFlow Lite Setup ===
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
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (input_det["shape"][1], input_det["shape"][2]))
        if input_det["dtype"] == np.float32:
            image = image.astype(np.float32) / 127.5 - 1.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logging.error(f"Preprocess error: {e}")
        return None

def run_inference_on_frame(frame, interpreter, input_details, output_details):
    input_data = preprocess_frame(frame, input_details)
    if input_data is None:
        return "error"

    try:
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])
        score = output_data[0][0]
        class_label = "unacceptable" if score <= 0.50 else "pet_bottle"
        logging.info(f"Prediction score: {score:.4f}, label: {class_label}")
        return class_label
    except Exception as e:
        logging.error(f"Inference error: {e}")
        return "error"

# === Object Tracking with ROI and Inference ===
# === Object Tracking with ROI and Inference ===
cap = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

x1, y1, x2, y2 = 230, 40, 370, 180
next_object_id = 0
objects = {}
TRACK_TIME = 1  # seconds
model_path = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/models_train/old/mobilenet_v2_standard.tflite"

interpreter, input_details, output_details = setup_tensor(model_path)

prediction_done = False
prediction_label = ""
start_time = time.time()
camera_timeout = 5  # seconds
freeze_duration = 30  # How long to keep the foreground after prediction (in frames)
freeze_counter = 0  # Counter for freeze duration

def get_centroid(cnt):
    M = cv.moments(cnt)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

def match_existing_objects(centroid, threshold=20):
    for obj_id, data in objects.items():
        ox, oy = data['centroid']
        if abs(ox - centroid[0]) < threshold and abs(oy - centroid[1]) < threshold:
            return obj_id
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Timeout fallback to unacceptable
    if not prediction_done and (current_time - start_time > camera_timeout):
        prediction_label = "unacceptable"
        prediction_done = True
        logging.info("Timeout reached. Defaulted to: unacceptable")

    fgmask = fgbg.apply(frame)
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
    # fgmask = cv.dilate(fgmask, kernel, iterations=1)

    roi_mask = fgmask[y1:y2, x1:x2]
    contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    frame_with_tracking = frame.copy()
    cv.rectangle(frame_with_tracking, (x1, y1), (x2, y2), (0, 0, 255), 2)

    seen_ids = set()

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 300:
            continue

        cnt_offset = cnt + np.array([[x1, y1]])
        centroid = get_centroid(cnt_offset)
        if centroid is None:
            continue

        matched_id = match_existing_objects(centroid)
        if matched_id is not None:
            objects[matched_id]['centroid'] = centroid
            seen_ids.add(matched_id)
            if not objects[matched_id]['tracked']:
                if current_time - objects[matched_id]['start_time'] >= TRACK_TIME:
                    objects[matched_id]['tracked'] = True

                    # Run inference once when tracking completes
                    prediction_label = run_inference_on_frame(frame, interpreter, input_details, output_details)
                    logging.info(f"Object #{matched_id} classified as: {prediction_label}")
                    print(f"Object #{matched_id} classified as: {prediction_label}")
                    prediction_done = True
                    freeze_counter = 0  # Reset freeze counter

        else:
            objects[next_object_id] = {
                'centroid': centroid,
                'start_time': current_time,
                'tracked': False
            }
            seen_ids.add(next_object_id)
            next_object_id += 1

    objects = {obj_id: data for obj_id, data in objects.items() if obj_id in seen_ids}

    for obj_id, data in objects.items():
        cx, cy = data['centroid']
        if data['tracked']:
            cv.circle(frame_with_tracking, (cx, cy), 5, (0, 255, 0), -1)
            cv.putText(frame_with_tracking, f"Tracked #{obj_id}", (cx + 10, cy),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            remaining = int(TRACK_TIME - (current_time - data['start_time']))
            cv.putText(frame_with_tracking, f"{remaining}s", (cx + 10, cy),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Show prediction result on frame if done
    if prediction_done:
        freeze_counter += 1
        cv.putText(frame_with_tracking, f"Prediction: {prediction_label}", (30, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if freeze_counter >= freeze_duration:
            prediction_done = False  # Reset prediction after freeze duration ends

    cv.imshow("Foreground Mask", fgmask)
    cv.imshow("ROI Mask", roi_mask)
    cv.imshow("Tracking + Inference", frame_with_tracking)

    if prediction_done:
        cv.waitKey(1000)  # Wait 1 second to show prediction
        break

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
