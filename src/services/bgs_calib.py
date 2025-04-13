
import time
from collections import defaultdict

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)

# Create trackbars for controlling the morphological operation kernel shape and size
def nothing(x):
    pass

def main():
    # Create windows for the trackbars
    cv.namedWindow("Control Panel")

    # Create trackbar for selecting the kernel size
    cv.createTrackbar("Kernel Size", "Control Panel", 5, 20, nothing)

    # Create trackbar for selecting morphological shape
    cv.createTrackbar("Element Shape", "Control Panel", 0, 2, nothing)  # 0: Rect, 1: Cross, 2: Ellipse

    kernel_size = 5  # Default kernel size
    morph_shape = cv.MORPH_RECT  # Default morphological shape (rectangle)

    # ROI Coordinates
    x1, y1, x2, y2 = 230, 40, 370, 180

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current kernel size and morphological shape from trackbars
        kernel_size = cv.getTrackbarPos("Kernel Size", "Control Panel")
        morph_shape = cv.getTrackbarPos("Element Shape", "Control Panel")

        # Print the trackbar positions (useful for experimentation)
        print(f"Kernel Size: {kernel_size} | Element Shape: {morph_shape}")

        # Apply Background Subtraction
        fgmask = fgbg.apply(frame)

        # Map the element shape value to the appropriate constant for morphological operation
        if morph_shape == 0:
            element = cv.MORPH_RECT
        elif morph_shape == 1:
            element = cv.MORPH_CROSS
        elif morph_shape == 2:
            element = cv.MORPH_ELLIPSE

        # Create a structuring element with the selected kernel size
        kernel = cv.getStructuringElement(element, (2 * kernel_size + 1, 2 * kernel_size + 1))

        # Apply Morphology (you can experiment with these operations)
        # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        # fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
        # fgmask = cv.erode(fgmask, kernel, iterations=1)
        # fgmask = cv.dilate(fgmask, kernel, iterations=1)  # Dilate -> Thicken white ; Erosion -> remove white

        # Extract ROI
        roi_mask = fgmask[y1:y2, x1:x2]

        # Show the results
        cv.imshow("ROI Mask", roi_mask)
        cv.imshow("Frame", frame)
        cv.imshow("Foreground Mask", fgmask)

        # Exit the loop when the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
