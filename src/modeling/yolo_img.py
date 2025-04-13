from pathlib import Path

from ultralytics import YOLO

dataset_dir = Path("/home/untitled/Documents/Coding_Repository/python_journey/Capstone/waste-classification/data/for_yolo")

assert (dataset_dir / "train").exists(), "Missing train/ directory"
assert (dataset_dir / "val").exists(), "Missing val/ directory"

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=str(dataset_dir),
    epochs=15,
    imgsz=224,
    batch=8,
    name="pet_bottle_classifier",
    patience=5,
    lr0=0.0001,
    lrf=0.01
)
