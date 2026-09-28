import sys
import cv2
from ultralytics import YOLO

sys.path.insert(0, "vision")
sys.path.insert(0, ".")

from component_fusion_engine import fuse_yolo_results
from engineering_state import EngineeringStateEngine

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

engine = EngineeringStateEngine()

engine.update_frame()

for track_id, item in enumerate(fused, start=1):
    engine.update_fused(
        track_id,
        item
    )

engine.update_lifecycle()

print()
print("SNAPLAB AI — ENGINEERING STATE FUSION")
print("=" * 70)

state = engine.get_state()

for track_id, component in state.items():
    print()
    print("TRACK ID:", track_id)
    print("FINAL CLASS:", component["class_name"])
    print("FINAL CONFIDENCE:", f"{component['confidence']:.4f}")
    print("SOURCE:", component["source"])
    print("STATUS:", component["status"])
    print("YOLO CLASS:", component["detector_class"])
    print(
        "YOLO CONFIDENCE:",
        f"{component['detector_confidence']:.4f}"
    )

    if component["classifier_class"] is not None:
        print(
            "MOBILENET CLASS:",
            component["classifier_class"]
        )
        print(
            "MOBILENET CONFIDENCE:",
            f"{component['classifier_confidence']:.4f}"
        )

    print("CENTER:", component["center"])
    print("BBOX:", component["bbox"])

print()
print("=" * 70)
print("COMPONENTS IN ENGINEERING STATE:", len(state))
print("ENGINEERING STATE TEST COMPLETE")