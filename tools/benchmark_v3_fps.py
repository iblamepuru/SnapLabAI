from ultralytics import YOLO
import cv2
import time

MODEL = r"runs\detect\runs\snaplab\phaseC_master_65class-2\weights\best.pt"

model = YOLO(MODEL)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

print("=" * 50)
print("SNAPLAB V3 FPS BENCHMARK")
print("=" * 50)

for imgsz in [640, 416, 320]:

    print(f"\nTesting imgsz={imgsz}...")

    # Warm-up
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            model.predict(
                frame,
                imgsz=imgsz,
                conf=0.20,
                verbose=False
            )

    # Benchmark
    start = time.perf_counter()
    count = 0

    for _ in range(30):
        ret, frame = cap.read()

        if not ret:
            continue

        model.predict(
            frame,
            imgsz=imgsz,
            conf=0.20,
            verbose=False
        )

        count += 1

    elapsed = time.perf_counter() - start
    fps = count / elapsed
    ms = (elapsed / count) * 1000

    print(f"  FPS       : {fps:.2f}")
    print(f"  Latency   : {ms:.2f} ms/frame")

cap.release()

print("\n" + "=" * 50)
print("BENCHMARK COMPLETE")
print("=" * 50)
