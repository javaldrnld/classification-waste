from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PETBottleDetectionUI(QMainWindow):
    def __init__(self, detection_logic):
        super().__init__()
        
        # Store reference to the logic layer
        self.logic = detection_logic
        
        # Connect signals from logic to UI update methods
        self.logic.new_frame_signal.connect(self.update_frame_display)
        self.logic.result_signal.connect(self.update_result_display)
        
        # Set up window properties
        self.setWindowTitle("PET Bottle Detection System")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize UI components
        self.init_ui()
        
    def init_ui(self):
        """Initialize all UI components"""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create frame for camera view
        self.camera_frame = QLabel()
        self.camera_frame.setFixedSize(self.logic.frame_width, self.logic.frame_height)
        self.camera_frame.setFrameShape(QFrame.Box)
        self.camera_frame.setAlignment(Qt.AlignCenter)
        self.camera_frame.setText("Camera Feed Will Appear Here")
        main_layout.addWidget(self.camera_frame, alignment=Qt.AlignCenter)
        
        # Create button to start/stop the system
        self.start_button = QPushButton("Insert PET Bottle")
        self.start_button.setMinimumSize(150, 50)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px;")
        self.start_button.clicked.connect(self.toggle_camera)
        main_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        
        # Create label for displaying detection results
        self.result_label = QLabel("System Ready")
        self.result_label.setMinimumSize(300, 50)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: #E0E0E0; font-size: 14px; border-radius: 5px;")
        main_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
    
    def toggle_camera(self):
        """Handle camera toggle button click"""
        if not self.logic.camera_active:
            # Start the camera
            self.logic.start_camera()
            self.start_button.setText("Stop System")
            self.start_button.setStyleSheet("background-color: #F44336; color: white; font-size: 14px;")
        else:
            # Stop the camera
            self.logic.stop_camera()
            self.start_button.setText("Insert PET Bottle")
            self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px;")
            self.result_label.setText("System Ready")
            self.result_label.setStyleSheet("background-color: #E0E0E0; font-size: 14px; border-radius: 5px;")
            self.camera_frame.setText("Camera Feed Will Appear Here")
            
    @pyqtSlot(QImage)
    def update_frame_display(self, image):
        """Update the camera frame with the provided QImage"""
        self.camera_frame.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot(str, str)
    def update_result_display(self, text, color):
        """Update the result label with text and color"""
        self.result_label.setText(text)
        self.result_label.setStyleSheet(f"background-color: {color}; font-size: 14px; border-radius: 5px; color: white;")
    
    def closeEvent(self, event):
        """Handle window closing event"""
        self.logic.cleanup()
        event.accept()