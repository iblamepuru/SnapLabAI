from ultralytics import YOLO
import cv2
import time

# ============================================================
# SNAPLAB AI — V3 REAL-TIME ENGINEERING VISION
# Snapdragon-ready baseline
# ============================================================

MODEL = r"runs\detect\runs\snaplab\phaseC_master_65class-2\weights\best.pt"

IMG_SIZE = 640
CONFIDENCE = 0.20

model = YOLO(MODEL)

print("=" * 60)
print("SNAPLAB AI — V3 REAL-TIME VISION")
print("=" * 60)
print("Model      : YOLO11n")
print("Classes    : 65")
print("Input      : 640x640")
print("Confidence : 0.20")
print("Backend    : CPU prototype / Snapdragon-ready")
print("=" * 60)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS measurement
fps_timer = time.perf_counter()
frame_counter = 0
fps = 0.0

# Inference measurement
inference_ms = 0.0

print("Camera started.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame read failed.")
        break

    # --------------------------------------------------------
    # AI inference
    # --------------------------------------------------------

    start = time.perf_counter()

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False
    )

    inference_ms = (time.perf_counter() - start) * 1000

    result = results[0]

    # --------------------------------------------------------
    # Render detections
    # --------------------------------------------------------

    annotated = result.plot()

    detection_count = 0

    if result.boxes is not None:
        detection_count = len(result.boxes)

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    frame_counter += 1

    elapsed = time.perf_counter() - fps_timer

    if elapsed >= 1.0:
        fps = frame_counter / elapsed
        frame_counter = 0
        fps_timer = time.perf_counter()

    # --------------------------------------------------------
    # Performance dashboard
    # --------------------------------------------------------

    cv2.rectangle(
        annotated,
        (0, 0),
        (325, 115),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated,
        "SNAPLAB AI - V3",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"AI: {inference_ms:.1f} ms",
        (10, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Objects: {detection_count}",
        (10, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "SnapLab AI - Snapdragon Engineering Copilot",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("SNAPLAB V3 TEST COMPLETED")
print("=" * 60)
print(f"Last measured inference : {inference_ms:.2f} ms")
print(f"Measured FPS             : {fps:.2f}")
print("=" * 60)
