import sys
import cv2
from ultralytics import YOLO

sys.path.insert(0, "vision")

from component_fusion_engine import fuse_yolo_results

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

fused = fuse_yolo_results(
    image,
    result,
    refinement_threshold=0.50
)

print()
print("SNAPLAB AI — YOLO + MOBILENET FUSION V2")
print("=" * 70)

for item in fused:
    print()
    print("FINAL CLASS:", item["final_class"])
    print("FINAL CONFIDENCE:", f"{item['final_confidence']:.4f}")
    print("SOURCE:", item["source"])
    print("STATUS:", item["status"])
    print("YOLO CLASS:", item["detector_class"])
    print("YOLO CONFIDENCE:", f"{item['detector_confidence']:.4f}")

    if item["classifier_class"] is not None:
        print("MOBILENET CLASS:", item["classifier_class"])
        print(
            "MOBILENET CONFIDENCE:",
            f"{item['classifier_confidence']:.4f}"
        )

print()
print("=" * 70)
print("TOTAL DETECTIONS:", len(fused))
print("FUSION TEST COMPLETE")