from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # small model, fast for CPU

# Run detection on webcam (0 = default laptop webcam)
model.predict(source=0, show=True, save=True)
