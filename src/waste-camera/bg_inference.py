import time
from collections import defaultdict

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# ROI
x1, y1, x2, y2 = 230, 40, 370, 180

# Object tracking
next_object_id = 0
objects = {}  # object_id: {'centroid': (x, y), 'start_time': float, 'tracked': bool}
TRACK_TIME = 2  # seconds

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

    fgmask = fgbg.apply(frame)
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
    fgmask = cv.dilate(fgmask, kernel, iterations=1)

    roi_mask = fgmask[y1:y2, x1:x2]
    contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    frame_with_tracking = frame.copy()
    cv.rectangle(frame_with_tracking, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Keep track of current object IDs seen in this frame
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
            # Update existing object
            objects[matched_id]['centroid'] = centroid
            seen_ids.add(matched_id)
            if not objects[matched_id]['tracked']:
                if current_time - objects[matched_id]['start_time'] >= TRACK_TIME:
                    objects[matched_id]['tracked'] = True
        else:
            # Register new object
            objects[next_object_id] = {
                'centroid': centroid,
                'start_time': current_time,
                'tracked': False
            }
            seen_ids.add(next_object_id)
            next_object_id += 1

    # Remove objects that are no longer in view
    objects = {obj_id: data for obj_id, data in objects.items() if obj_id in seen_ids}

    # Draw tracked objects
    for obj_id, data in objects.items():
        cx, cy = data['centroid']
        if data['tracked']:
            cv.circle(frame_with_tracking, (cx, cy), 5, (0, 255, 0), -1)
            cv.putText(frame_with_tracking, f"Tracked #{obj_id}", (cx + 10, cy),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("INFERENCE RUN HERE")
        else:
            remaining = int(TRACK_TIME - (current_time - data['start_time']))
            cv.putText(frame_with_tracking, f"{remaining}s", (cx + 10, cy),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Show output
    cv.imshow("Foreground Mask", fgmask)
    cv.imshow("ROI Mask", roi_mask)
    cv.imshow("Tracking in ROI", frame_with_tracking)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
