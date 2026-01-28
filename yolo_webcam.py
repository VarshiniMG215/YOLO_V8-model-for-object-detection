from ultralytics import YOLO
import cv2

# 1) load a tiny pre-trained model
model = YOLO("yolov8n.pt")  # downloads on first run

# 2) open webcam (try 0,1,2 if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing the index (0→1→2).")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # 3) run detection
    results = model(frame)        # by default runs on CPU; on GPU if available
    annotated = results[0].plot() # draw boxes & labels

    # 4) show
    cv2.imshow("YOLOv8 Webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
