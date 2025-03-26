import sys

from logic.detection_logic import DetectionLogic
from PyQt5.QtWidgets import QApplication
from ui.main_window import PETBottleDetectionUI


def main():
    app = QApplication(sys.argv)
    
    # Pass model path to DetectionLogic
    detection_logic = DetectionLogic(model_path="../models_train/transfer_learningv1.tflite")
    
    ui = PETBottleDetectionUI(detection_logic)
    ui.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
