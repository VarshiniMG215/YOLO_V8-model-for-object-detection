from ultralytics import YOLO
import cv2

# classes we consider "obstacles" for a quick demo
OBSTACLE_CLASS_NAMES = {"person","bicycle","car","motorbike","bus","truck","bench","chair","dog","cat","bird"}

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

def in_center(bbox, frame_w, frame_h, center_ratio=0.4):
    cx1 = int((1-center_ratio)/2 * frame_w)
    cy1 = int((1-center_ratio)/2 * frame_h)
    cx2 = int((1+(center_ratio))/2 * frame_w)
    cy2 = int((1+(center_ratio))/2 * frame_h)
    x1,y1,x2,y2 = map(int, bbox)
    bx = (x1+x2)//2
    by = (y1+y2)//2
    return (cx1 <= bx <= cx2) and (cy1 <= by <= cy2), (cx1,cy1,cx2,cy2)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    _, (cx1,cy1,cx2,cy2) = in_center([0,0,0,0], w, h)
    cv2.rectangle(annotated, (cx1,cy1), (cx2,cy2), (0,255,255), 2)

    stop = False
    for b in results[0].boxes:
        cls_id = int(b.cls[0])
        name = results[0].names[cls_id]
        bbox = b.xyxy[0].tolist()
        inside, _ = in_center(bbox, w, h)
        if name in OBSTACLE_CLASS_NAMES and inside:
            stop = True
            x1,y1,x2,y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 3)

    text = "STOP" if stop else "GO"
    cv2.putText(annotated, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0,0,255) if stop else (0,255,0), 3)
    cv2.imshow("YOLOv8 STOP/GO", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
