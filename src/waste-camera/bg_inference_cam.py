import time
from collections import defaultdict

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

""" Notes
 - If varying 'yong light, best na gawing adaptive, then add ng learning rate sa MOG2 
    - Lower learning rate: mas
 - Histogram Equalization (CLAHE)
"""
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# ROI -> Can be changed soon depends sa location or angle
time_threshold = 1.0 # Time that object should be in second
x1, y1, x2, y2 = 230, 40, 370, 180
roi_area_threshold = 500 # Minimum contour area in ROI para maconsider as detected
frames_threshold = int(time_threshold * cap.get(cv.CAP_PROP_FPS)) # Convert to frames 
frame_counter = 0 # Counter for frames with object present
detection_triggered = False
camera_timeout = 50.0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_with_roi = frame.copy()


    # check for timeout
    if not detection_triggered and (current_time - start_time > camera_timeout):
        prediction_label = "unacceptable"
        detection_triggered = True
        cv.putText(frame_with_roi, "Prediction: Wala", (30, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # Apply Morphology to remove noise and then apply dilate to thicken the white 
    # Currently naka comment since nagca-cause ng disruption or nawawala bigla
    fgmask = fgbg.apply(frame)
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
    # fgmask = cv.erode(fgmask, kernel, iterations=1)
    # fgmask = cv.dilate(fgmask, kernel, iterations=1) # Dilate -> Thicken white ; Erosion -> remove white

    # Instead of contour frame, focus only on the pixel or centroid of the frame using the fgmask
    # Get the width of y1, y2, as well as height of x1, x2
    roi_mask = fgmask[y1:y2, x1:x2]
    
    """ Notes
    findContours (image, mode, method)
    - Mode: How contours are organized and retrieved -> which contours to be detected in the image
        - RETR_EXTERNAL -> Retrieves external contours
        - RETR_LIST -> All contours but does not create hierarchy beteen them
        - RETR_TREE -> Retrieves all contours and reconstrucs a full hierachy of nested contour
        - RETR_COMP -> Retrieves all the contours and organize into two levels
        - RETR_FLOODFILL -> Retrives all contours and trates the image like a flood-filled image

    - Method:  How the contours are approx
        - CHAIN_APPROX_SIMPLE -> Removes all redundant points and compresses the contour, storing only the essential points
        - CHAIN_APPROX_NONE -> Stores all the points of the contour 
    """
    contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Calculate total area relating sa roi_mask then don mag de-decide if capture or continue
    total_area_roi = 0
    for cnt in contours:
        # Dito check lang 'yong contour sa area (roi_mask) then add sa total_area if mag s-stay or papasok sa limit
        # Each moment to ng frame
        area = cv.contourArea(cnt)
        print(f"Area of contour: {area}")
        if area > 300: # P'wedeng baguhin pa 'to for calibration na tatanggapin
            total_area_roi += area
            print(f"Area that surpass filter: {area}")
            print(f"Total Area: {total_area_roi}")

    print(f"Total outside: {total_area_roi}")
    
    # Display frame with ROI
    cv.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0,0,255), 2)
    
    # Object persistence check
    if total_area_roi > roi_area_threshold:
        frame_counter += 1
        # Show counter on frame
        cv.putText(frame_with_roi, f"Detecting: {frame_counter}/{frames_threshold}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Check if object is nandon for 1 second (30 frames) adjustable
        if frame_counter >= frames_threshold and not detection_triggered:
            # Run inference here
            print(f"Object classified as: prediction label")
            # Reset the detection_triggered
            detection_triggered = True
            print("HINTO")
    else:
        # Reset counter if nawala agad 
        frame_counter = 0
        cv.putText(frame_with_roi, "Waiting for object...", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    if detection_triggered:
        cv.putText(frame_with_roi, "Prediction: MJ GOAT!!!", (30, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv.imshow("ROI Mask", roi_mask)
    cv.imshow("Frame", frame_with_roi)
    # cv.imshow("Foreground Mask", fgmask)


    if detection_triggered:
        cv.waitKey(1000)
        break
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
#cv.destroyAllWindows()
