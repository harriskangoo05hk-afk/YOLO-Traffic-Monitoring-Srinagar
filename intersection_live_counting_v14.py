import cv2
import time
import os
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------------- SETTINGS ----------------

FRAME_SKIP = 6  # Skip frames to dramatically speed up YOLO processing
MIN_GREEN = 15
MAX_GREEN = 60
YELLOW_TIME = 3

FRAME_SIZE = (720, 720)

# ---------------- OUTPUT ----------------

os.makedirs("output_videos", exist_ok=True)
os.makedirs("output_data", exist_ok=True)

# ---------------- YOLO ----------------

model = YOLO("yolov8n.pt")
ALLOWED = {"car","motorcycle","bus","truck"}

# ---------------- INPUT ----------------

cap1 = cv2.VideoCapture("Section1.mp4")
cap2 = cv2.VideoCapture("Section2.mp4")
cap3 = cv2.VideoCapture("Section3.mp4")

# ---------------- VIDEO OUTPUT ----------------

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out1 = cv2.VideoWriter("output_videos/sec1.mp4", fourcc, 20, FRAME_SIZE)
out2 = cv2.VideoWriter("output_videos/sec2.mp4", fourcc, 20, FRAME_SIZE)
out3 = cv2.VideoWriter("output_videos/sec3.mp4", fourcc, 20, FRAME_SIZE)

# ---------------- PCU ----------------

PCU = {"car":1, "motorcycle":0.5, "bus":3, "truck":3}

# ---------------- ZONES ----------------

ZONE1 = (300,200,900,500)
ZONE2 = (300,200,900,500)
ZONE3 = (300,200,900,500)

# ---------------- STATE ----------------

frame_count = 0
cycle = 0
prev_cycle_length = 60.0  # Assumed initial cycle length

last_boxes1 = []
last_boxes2 = []
last_boxes3 = []

# ---------------- DATA STORAGE ----------------

log_data = []

# ------------------------------------------------------------

def compute_pcu(frame, zone):

    x1,y1,x2,y2 = zone
    display = frame.copy()

    cv2.rectangle(display,(x1,y1),(x2,y2),(255,255,0),2)

    # Use smaller imgsz and hide verbose logs to speed up FPS significantly
    results = model(frame, verbose=False, imgsz=480)[0]
    total = 0
    boxes_to_keep = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label not in ALLOWED:
            continue

        bx1,by1,bx2,by2 = map(int, box.xyxy[0])
        cx = (bx1+bx2)//2
        cy = (by1+by2)//2

        if x1 < cx < x2 and y1 < cy < y2:
            total += PCU[label]
            color = (0,255,0)
            thickness = 2
        else:
            color = (0,0,255)
            thickness = 1
            
        cv2.rectangle(display,(bx1,by1),(bx2,by2),color,thickness)
        boxes_to_keep.append((bx1,by1,bx2,by2,color,thickness))

    return total, display, boxes_to_keep


def draw_cached_boxes(frame, zone, boxes):
    display = frame.copy()
    x1,y1,x2,y2 = zone
    cv2.rectangle(display,(x1,y1),(x2,y2),(255,255,0),2)
    for bx1,by1,bx2,by2,color,thickness in boxes:
        cv2.rectangle(display,(bx1,by1),(bx2,by2),color,thickness)
    return display


def webster_green_times(pcus, prev_cycle_length=60.0):
    # Convert instantaneous PCU queue (density) into an empirical Flow Rate (q).
    # Since these vehicles accumulated roughly over the previous cycle length (T_prev),
    # the arrival flow rate q = PCU / T_prev (in PCU per second).
    # Saturation flow limit S = 1800 PCU/hr = 0.5 PCU/sec.
    # Therefore, Flow Ratio y = q / S = (PCU / T_prev) / 0.5 = (2 * PCU) / T_prev
    
    # +1 ensures a minimum y to prevent division errors and allocate base green time
    y = [(2.0 * (p + 1)) / prev_cycle_length for p in pcus]
    
    Y = sum(y)
    Y = min(float(Y), 0.85)  # Cap flow ratio sum to avoid infinite cycle times
    
    # Lost time L = n*(startup_lost + yellow)
    # n=3 phases, startup lost=2s, yellow=YELLOW_TIME
    L = 3 * (2 + YELLOW_TIME)
    
    # Optimum cycle length C0
    C0 = (1.5 * float(L) + 5.0) / (1.0 - Y)
    C0 = max(45.0, min(120.0, float(C0)))  # Constrain between 45s and 120s
    
    # Effective green time to distribute
    total_effective_green = C0 - L
    
    # Allocate proportional to y_i
    g_times = [(yi / Y) * total_effective_green for yi in y]
    
    return [max(MIN_GREEN, int(g)) for g in g_times]


def apply_signal(f1, f2, f3, chosen, mode):

    colors = {
        "GREEN": (0,255,0),
        "RED": (0,0,255),
        "YELLOW": (0,255,255)
    }

    if chosen == 0:
        cv2.putText(f1,f"S1 {mode}",(50,80),0,1,colors[mode],3)
        cv2.putText(f2,"S2 RED",(50,80),0,1,colors["RED"],3)
        cv2.putText(f3,"S3 RED",(50,80),0,1,colors["RED"],3)

    elif chosen == 1:
        cv2.putText(f2,f"S2 {mode}",(50,80),0,1,colors[mode],3)
        cv2.putText(f1,"S1 RED",(50,80),0,1,colors["RED"],3)
        cv2.putText(f3,"S3 RED",(50,80),0,1,colors["RED"],3)

    else:
        cv2.putText(f3,f"S3 {mode}",(50,80),0,1,colors[mode],3)
        cv2.putText(f1,"S1 RED",(50,80),0,1,colors["RED"],3)
        cv2.putText(f2,"S2 RED",(50,80),0,1,colors["RED"],3)


def run_phase(duration, chosen, mode):

    global frame_count, last_boxes1, last_boxes2, last_boxes3

    start = time.time()

    while time.time() - start < duration:

        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        ret3, f3 = cap3.read()

        if not ret1 or not ret2 or not ret3:
            return False

        f1 = cv2.resize(f1, FRAME_SIZE)
        f2 = cv2.resize(f2, FRAME_SIZE)
        f3 = cv2.resize(f3, FRAME_SIZE)

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            _, f1, last_boxes1 = compute_pcu(f1, ZONE1)
            _, f2, last_boxes2 = compute_pcu(f2, ZONE2)
            _, f3, last_boxes3 = compute_pcu(f3, ZONE3)
        else:
            f1 = draw_cached_boxes(f1, ZONE1, last_boxes1)
            f2 = draw_cached_boxes(f2, ZONE2, last_boxes2)
            f3 = draw_cached_boxes(f3, ZONE3, last_boxes3)

        apply_signal(f1, f2, f3, chosen, mode)

        cv2.imshow("S1", f1)
        cv2.imshow("S2", f2)
        cv2.imshow("S3", f3)

        out1.write(f1)
        out2.write(f2)
        out3.write(f3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return True


print("SYSTEM RUNNING WITH DATA LOGGING")

while True:

    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    ret3, f3 = cap3.read()

    if not ret1 or not ret2 or not ret3:
        break

    f1 = cv2.resize(f1, FRAME_SIZE)
    f2 = cv2.resize(f2, FRAME_SIZE)
    f3 = cv2.resize(f3, FRAME_SIZE)

    p1, _, last_boxes1 = compute_pcu(f1, ZONE1)
    p2, _, last_boxes2 = compute_pcu(f2, ZONE2)
    p3, _, last_boxes3 = compute_pcu(f3, ZONE3)

    pcus = [p1,p2,p3]

    cycle += 1

    # Webster Sequence logic
    g_times = webster_green_times(pcus, prev_cycle_length)
    
    # Update prev_cycle_length for the next calculation
    # Current cycle length = Sum of all effective green times + 3 yellow times
    prev_cycle_length = sum(g_times) + 3.0 * YELLOW_TIME

    # Store cycle data: [Cycle, PCU1, PCU2, PCU3, G1, G2, G3]
    log_data.append([cycle, p1, p2, p3, g_times[0], g_times[1], g_times[2]])

    # Run all three phases sequentially for this cycle
    running = True
    for chosen in range(3):
        if not run_phase(g_times[chosen], chosen, "GREEN"):
            running = False
            break
        if not run_phase(YELLOW_TIME, chosen, "YELLOW"):
            running = False
            break
            
    if not running:
        break

# ---------------- SAVE CSV ----------------

with open("output_data/traffic_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Cycle","PCU1","PCU2","PCU3","Green1","Green2","Green3"])
    writer.writerows(log_data)

# ---------------- GRAPHS ----------------

cycles = [row[0] for row in log_data]
pcu1 = [row[1] for row in log_data]
pcu2 = [row[2] for row in log_data]
pcu3 = [row[3] for row in log_data]
g1 = [row[4] for row in log_data]
g2 = [row[5] for row in log_data]
g3 = [row[6] for row in log_data]

plt.figure()
plt.plot(cycles, pcu1, marker='o')
plt.plot(cycles, pcu2, marker='s')
plt.plot(cycles, pcu3, marker='^')
plt.title("PCU vs Cycle")
plt.xlabel("Cycle")
plt.ylabel("PCU")
plt.legend(["S1","S2","S3"])
plt.savefig("output_data/pcu_graph.png")

plt.figure()
plt.plot(cycles, g1, marker='o')
plt.plot(cycles, g2, marker='s')
plt.plot(cycles, g3, marker='^')
plt.title("Webster Green Time vs Cycle")
plt.xlabel("Cycle")
plt.ylabel("Green Time (s)")
plt.legend(["S1", "S2", "S3"])
plt.savefig("output_data/green_time.png")

print("CSV and graphs saved in output_data folder")

# ---------------- CLEANUP ----------------

cap1.release()
cap2.release()
cap3.release()
out1.release()
out2.release()
out3.release()
cv2.destroyAllWindows()