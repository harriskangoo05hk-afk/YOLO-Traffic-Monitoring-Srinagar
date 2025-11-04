from ultralytics import YOLO
import cv2
import time
import csv
import os
from collections import defaultdict

# ✅ Camera facing Lal Chowk (West)
CAMERA_FACING_WEST = True

# ✅ Load YOLO model
model = YOLO("yolov8n.pt")

# ✅ Create save folder
save_folder = "results"
os.makedirs(save_folder, exist_ok=True)

# ✅ Open laptop webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# ✅ Counting lines (adjust at site if needed)
vertical_line_x = frame_width // 2       # Middle of screen
horizontal_line_y = frame_height - 200   # For Residency Road flow

# ✅ Allowed vehicles
vehicle_classes = ["car", "truck", "bus", "motorcycle"]

# ✅ Tracking and counting storage
vehicle_tracks = {}
counts = defaultdict(int)

# ✅ Save data to CSV
csv_filename = os.path.join(save_folder, "traffic_counts.csv")
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Direction", "Vehicle Type"])

print("✅ Traffic Counter Started — Press Q to Stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            label = model.names[cls]

            if label not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            track_id = id(box)
            if track_id not in vehicle_tracks:
                vehicle_tracks[track_id] = {"centroids": [], "counted": False}

            vehicle_tracks[track_id]["centroids"].append((cx, cy))
            pts = vehicle_tracks[track_id]["centroids"]

            if len(pts) > 2 and not vehicle_tracks[track_id]["counted"]:
                (px, py) = pts[-2]
                (nx, ny) = pts[-1]

                # ✅ Movement M3: Residency Road → Dalgate
                if py > horizontal_line_y and ny <= horizontal_line_y:
                    counts["Residency→Dalgate"] += 1
                    vehicle_tracks[track_id]["counted"] = True
                    with open(csv_filename, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [time.strftime("%H:%M:%S"), "Residency→Dalgate", label])

                else:
                    # ✅ Vertical line logic (depends on camera facing)
                    if CAMERA_FACING_WEST:
                        # Left→Right = Dalgate→Lal Chowk
                        if px < vertical_line_x and nx >= vertical_line_x:
                            counts["Dalgate→LalChowk"] += 1
                            vehicle_tracks[track_id]["counted"] = True
                            with open(csv_filename, "a", newline="") as f:
                                csv.writer(f).writerow(
                                    [time.strftime("%H:%M:%S"), "Dalgate→LalChowk", label])

                        # Right→Left = Lal Chowk→Dalgate
                        elif px > vertical_line_x and nx <= vertical_line_x:
                            counts["LalChowk→Dalgate"] += 1
                            vehicle_tracks[track_id]["counted"] = True
                            with open(csv_filename, "a", newline="") as f:
                                csv.writer(f).writerow(
                                    [time.strftime("%H:%M:%S"), "LalChowk→Dalgate", label])

            # ✅ Draw detection box + points
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    # ✅ Draw counting lines
    cv2.line(frame, (vertical_line_x, 0), (vertical_line_x, frame_height), (0, 255, 0), 2)
    cv2.line(frame, (0, horizontal_line_y), (frame_width, horizontal_line_y), (255, 0, 0), 2)

    # ✅ Display live counters
    cv2.putText(frame, f"Lal Chowk → Dalgate: {counts['LalChowk→Dalgate']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Dalgate → Lal Chowk: {counts['Dalgate→LalChowk']}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Residency → Dalgate: {counts['Residency→Dalgate']}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Intersection Traffic Counter", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done! Data saved at:", csv_filename)
