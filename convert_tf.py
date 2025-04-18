from ultralytics import YOLO
model = YOLO("/home/untitled/Documents/Coding_Repository/python_journey/Capstone/waste-classification/models_train/n/weights/best.pt")

model.export(format="tflite", imgsz=224, project='/app/models', name='v8_18_25_classification_n')