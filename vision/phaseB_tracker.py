from ultralytics import YOLO
import cv2

MODEL = r"runs\detect\runs\snaplab\phaseB_61class\weights\best.pt"

model = YOLO(MODEL)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("SnapLab-AI - 61-Class Object Tracker")
print("Tracking: ByteTrack")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame read failed.")
        break

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.40,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow(
        "SnapLab-AI - Electronics Tracker",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
