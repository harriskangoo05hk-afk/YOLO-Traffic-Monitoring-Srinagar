🚦 YOLO-Traffic-Monitoring-Srinagar
Real-Time Vehicle Detection & Direction-wise Counting at MA Road – Lal Chowk T-Intersection

This project uses YOLO (You Only Look Once) and Computer Vision to detect vehicles in real-time and count them based on their movement direction.
It is designed for the busy T-intersection at Lal Chowk, Srinagar (MA Road – Dalgate – Residency Road).

✅ Features Implemented

Live video feed from webcam
YOLO-based vehicle detection (Car, Bus, Truck, Motorcycle)
Unique tracking IDs to avoid double counting

Direction-based classification:
Lal Chowk → Dalgate
Dalgate → Lal Chowk
Residency Road → Dalgate

Real-time count visualization
Automatic saving of:
Annotated video output
CSV logs of vehicle counts

Prototype successfully tested at home and ready for on-site field trial at footbridge on MA Road.

🎯 Objective
To support intelligent traffic management by:
1️⃣ Analyzing real vehicle movement behavior
2️⃣ Understanding direction-based peak flow patterns
3️⃣ Enabling future adaptive signal optimization

🧠 Model Used
YOLOv8n (Ultralytics)
CPU execution supported (no GPU required)

🖥️ How to Run
Install Requirements
pip install ultralytics opencv-python numpy

Run Command
python intersection_live_counting.py


Press Q to stop video processing.

📂 Recommended Folder Structure
YOLO-Traffic-Monitoring-Srinagar/
├── code/
│   └── intersection_live_counting.py
├── results/
│   ├── videos/
│   └── logs/
└── README.md

📚 Literature Basis
Research Support	Reference
Real-time YOLO detection + tracking	Paper 1
Traffic signal optimization based on vehicle count	Paper 2
Direction-wise movement analysis	Paper 3

The ideas from these papers guided our design, objectives, and future scope.

📌 Next Work

🔹 Field test at MA Road footbridge
🔹 Improve detection line accuracy
🔹 Analyze peak hour flows
🔹 Integrate adaptive signal timing logic (Phase-2)

👨‍💻 Student Details

Harris Kangoo
Roll No: 2024MCIVTP012
M.Tech Transportation Engineering and Planning
National Institute of Technology Srinagar

🧑‍🏫 Guide

Dr. Janani L.
Assistant Professor
Department of Transportation Engineering & Planning
NIT Srinagar

✨ Acknowledgement

I would like to thank my guide Dr. Janani L. for constant support and guidance in this work.
