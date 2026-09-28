import sys
import cv2
from ultralytics import YOLO

sys.path.insert(0, "vision")

from component_refinement_engine import refine_detection

image = cv2.imread("vision/reference_frame.jpg")

model = YOLO(
    r"runs\detect\runs\snaplab\phaseC_master_65class-2\weights\best.pt"
)

result = model.predict(
    image,
    imgsz=640,
    conf=0.001,
    device="cpu",
    verbose=False
)[0]

found = False

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    if class_name == "Resistor":
        found = True

        print("YOLO DETECTION")
        print("Class:", class_name)
        print("Confidence:", f"{confidence:.4f}")

        refined = refine_detection(
            image,
            box.xyxy[0].tolist(),
            padding=35,
            scale=4.0
        )

        print()
        print("MOBILENET REFINEMENT")

        if refined is None:
            print("REFINEMENT: FAILED")
        else:
            print("Class:", refined["class_name"])
            print("Confidence:", f"{refined['confidence']:.4f}")
            print("Crop:", refined["crop"].shape)

        break

if not found:
    print("NO RESISTOR CANDIDATE FOUND")