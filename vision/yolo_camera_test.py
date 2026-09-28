import cv2
from ultralytics import YOLO

print("SNAPLAB AI — YOLO CAMERA TEST")
print("=" * 45)

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("ERROR: Could not open camera.")

print("Model       : YOLO11n")
print("Camera      : OK")
print("Inference   : CPU baseline")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Failed to read camera frame.")
        break

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.40,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.putText(
        annotated,
        "SnapLab AI - YOLO11n CPU Baseline",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("SnapLab AI - YOLO11n", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("YOLO CAMERA TEST COMPLETE")
