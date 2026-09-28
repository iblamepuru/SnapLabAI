import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wire_detector import WireDetector


detector = WireDetector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("SnapLab-AI - Wire Detection Camera Test")
print("Press Q to quit.")

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame read failed.")
        break

    wires = detector.detect(frame)

    output = frame.copy()

    for i, wire in enumerate(wires):

        x1, y1, x2, y2 = wire["bbox"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        label = (
            f'Wire {i + 1} | '
            f'L={wire["length"]} | '
            f'AR={wire["aspect_ratio"]}'
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    cv2.putText(
        output,
        f"Wire candidates: {len(wires)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SnapLab-AI - Wire Detector",
        output
    )

    frame_count += 1

    if frame_count % 30 == 0:
        print(
            f"Frame {frame_count} | "
            f"Wire candidates: {len(wires)}"
        )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Wire detection camera test completed.")
