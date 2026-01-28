# YOLO_V8-model-for-object-detection
#   we were working on AI-Driven Disaster Response Drone: Autonomous Navigation & Object Detection

This repository contains the intelligence system for a Disaster Response Drone designed to operate in high-risk environments. By leveraging a basic YOLOv8, the drone can identify victims, hazards, and infrastructure damage in real-time while performing autonomous obstacle avoidance.

##  Key Features
- Real-Time Object Detection: Uses YOLOv8 (specifically optimized for aerial views) to detect humans, fire, and debris.
- Collision Avoidance:Integrated obstacle detection logic to ensure safe flight through collapsed structures or forests.
- Search & Rescue Optimization:Lightweight architecture designed for edge computing on drones (e.g., Jetson Nano or OAK-D).
- Metric Extraction: Estimates distance to objects to help the drone calculate safe flight paths.

##  Technical Architecture
- Framework: PyTorch & Ultralytics
- Core Model: YOLOv8n (Nano) for 100+ FPS inference.
- Vision Library: OpenCV for image preprocessing and stream handling.
- Sensors Support: Designed for monocular camera feeds with depth-estimation potential.

##  Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/VarshiniMG215/DETECT_POTHOLE_Realtime.git](https://github.com/VarshiniMG215/DETECT_POTHOLE_Realtime.git)
