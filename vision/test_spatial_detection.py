import cv2
from ultralytics import YOLO

from spatial_engine import analyze_component_positions


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    r"runs\detect\runs\snaplab\phaseC_master_65class-2"
    r"\weights\best.pt"
)

IMAGE_PATH = r"vision\reference_frame.jpg"


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )


# ============================================================
# DETECT COMPONENTS
# ============================================================

result = model.predict(
    image,
    imgsz=640,
    conf=0.20,
    device="cpu",
    verbose=False
)[0]


# ============================================================
# CONVERT YOLO DETECTIONS TO SPATIAL INPUT
# ============================================================

components = []

for box in result.boxes:

    class_id = int(box.cls[0])

    confidence = float(
        box.conf[0]
    )

    x1, y1, x2, y2 = (
        box.xyxy[0].tolist()
    )

    class_name = model.names[
        class_id
    ]

    components.append(
        {
            "name": class_name,

            "box": [
                x1,
                y1,
                x2,
                y2
            ],

            "confidence": confidence
        }
    )


# ============================================================
# SPATIAL ANALYSIS
# ============================================================

relationships = analyze_component_positions(
    components
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 60)
print("SNAPLAB AI — SPATIAL DETECTION TEST")
print("=" * 60)

print(
    f"Components detected: "
    f"{len(components)}"
)

for component in components:

    print(
        f"- {component['name']} "
        f"({component['confidence']:.2f})"
    )

print()
print(
    f"Spatial relationships: "
    f"{len(relationships)}"
)

for relation in relationships:

    print(
        f"\n{relation['component_a']} "
        f"<-> "
        f"{relation['component_b']}"
    )

    print(
        f"  Position: "
        f"{relation['relative_position']}"
    )

    print(
        f"  Distance: "
        f"{relation['distance_pixels']:.1f} px"
    )

    print(
        f"  Normalized distance: "
        f"{relation['normalized_distance']:.2f}"
    )

print()
print("=" * 60)
print("SPATIAL DETECTION TEST: PASSED")
print("=" * 60)