import cv2
import time
import numpy as np

from snapdragon_npu_adapter import SnapdragonNPUAdapter
from yolo_decoder import decode_yolo_output

CONF = 0.20
IOU = 0.45
SIZE = 640

adapter = SnapdragonNPUAdapter()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

print("SNAPLAB AI - QUALCOMM MODEL CAMERA")
print("Press Q to quit.")

prev = time.perf_counter()

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    img = cv2.resize(frame, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = img.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    start = time.perf_counter()

    output = adapter.infer(tensor)

    inference_ms = (time.perf_counter() - start) * 1000.0

    detections = decode_yolo_output(
        output,
        conf_threshold=CONF,
        iou_threshold=IOU
    )

    display = cv2.resize(frame, (SIZE, SIZE))

    for d in detections:
        x, y, w, h = d["box"]

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(SIZE - 1, int(x + w))
        y2 = min(SIZE - 1, int(y + h))

        label = f'{d["class_name"]} {d["confidence"]:.2f}'

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            2
        )

        cv2.putText(
            display,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2
        )

    now = time.perf_counter()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now

    cv2.putText(
        display,
        "SNAPLAB AI - QUALCOMM MODEL",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Local inference: {inference_ms:.1f} ms",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Objects: {len(detections)}",
        (15, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imshow("SnapLab AI - Qualcomm Detection", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
