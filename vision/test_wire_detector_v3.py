import cv2
from ultralytics import YOLO

from wire_detector_v3 import (
    detect_wire_candidates,
    draw_wire_candidates
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"runs\detect\runs\snaplab\phaseC_master_65class-2"
    r"\weights\best.pt"
)

IMAGE_PATH = r"vision\reference_frame.jpg"

OUTPUT_PATH = r"vision\wire_detector_v3_debug.png"

CONFIDENCE = 0.20


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(
    MODEL_PATH
)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)

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
    conf=CONFIDENCE,
    device="cpu",
    verbose=False
)[0]


# ============================================================
# COLLECT COMPONENT BOXES
# ============================================================

component_boxes = []

components = []

for box in result.boxes:

    class_id = int(
        box.cls[0]
    )

    confidence = float(
        box.conf[0]
    )

    x1, y1, x2, y2 = (
        box.xyxy[0].tolist()
    )

    component_boxes.append(
        [
            x1,
            y1,
            x2,
            y2
        ]
    )

    components.append(
        {
            "name": model.names[class_id],
            "confidence": confidence,
            "box": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]
        }
    )


# ============================================================
# DETECT WIRE CANDIDATES
# ============================================================

raw_mask, clean_mask, candidates = (
    detect_wire_candidates(
        image,
        component_boxes=component_boxes,
        min_saturation=100,
        min_value=60,
        min_area=20,
        min_aspect_ratio=2.0
    )
)


# ============================================================
# DEBUG IMAGE
# ============================================================

debug_image = draw_wire_candidates(
    image,
    candidates
)


# Draw YOLO component boxes
for component in components:

    x1, y1, x2, y2 = (
        component["box"]
    )

    cv2.rectangle(
        debug_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        f"{component['name']} "
        f"{component['confidence']:.2f}"
    )

    cv2.putText(
        debug_image,
        label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


# Save debug image
cv2.imwrite(
    OUTPUT_PATH,
    debug_image
)


# ============================================================
# RESULTS
# ============================================================

print()

print(
    "=" * 60
)

print(
    "SNAPLAB AI — WIRE DETECTOR V3 TEST"
)

print(
    "=" * 60
)

print(
    f"Input image: {IMAGE_PATH}"
)

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
    f"Raw colored pixels: "
    f"{int((raw_mask > 0).sum())}"
)

print(
    f"Clean colored pixels: "
    f"{int((clean_mask > 0).sum())}"
)

print()

print(
    f"Wire candidates: "
    f"{len(candidates)}"
)

for index, candidate in enumerate(
    candidates[:20],
    start=1
):

    print(
        f"\nCandidate {index}"
    )

    print(
        f"  Box: "
        f"({candidate['x']}, "
        f"{candidate['y']}) → "
        f"({candidate['x'] + candidate['width']}, "
        f"{candidate['y'] + candidate['height']})"
    )

    print(
        f"  Size: "
        f"{candidate['width']} x "
        f"{candidate['height']} px"
    )

    print(
        f"  Area: "
        f"{candidate['area']} px"
    )

    print(
        f"  Aspect ratio: "
        f"{candidate['aspect_ratio']:.2f}"
    )


if len(candidates) > 20:

    print(
        f"\n... "
        f"{len(candidates) - 20} more candidates"
    )


print()

print(
    f"Debug image saved: "
    f"{OUTPUT_PATH}"
)

print()

print(
    "=" * 60
)

print(
    "WIRE DETECTOR V3 TEST: PASSED"
)

print(
    "=" * 60
)