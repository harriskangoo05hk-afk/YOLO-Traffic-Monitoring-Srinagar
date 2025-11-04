from ultralytics import YOLO
import cv2
import os
import glob

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")

# Vehicle classes to keep
allowed_classes = ["car", "bus", "truck", "motorcycle"]

# Source folder with images
source_folder = "C:/Users/USER/Desktop/vehicles_images"

# Output folder
output_folder = "C:/Users/USER/Desktop/YOLO/vehicle_results"
os.makedirs(output_folder, exist_ok=True)

# Get all image paths
image_paths = glob.glob(os.path.join(source_folder, "*.jpg"))  # use *.png if needed

# Process each image
for img_path in image_paths:
    img = cv2.imread(img_path)

    # Run detection
    results = model(img)  # pass image array directly

    # Draw only allowed vehicle boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in allowed_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save processed image
    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_folder, filename), img)
    print(f"Processed {filename}")
