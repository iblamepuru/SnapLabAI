import cv2
import numpy as np
from ultralytics import YOLO

from wire_detector import detect_wire_mask


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"runs\detect\runs\snaplab\phaseC_master_65class-2"
    r"\weights\best.pt"
)

IMAGE_PATH = r"vision\reference_frame.jpg"

OUTPUT_MASK_PATH = r"vision\wire_mask_v2_debug.png"

CONFIDENCE = 0.20

MIN_LINE_LENGTH = 20

MAX_LINE_GAP = 15


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


height, width = image.shape[:2]


# ============================================================
# YOLO COMPONENT DETECTION
# ============================================================

result = model.predict(
    image,
    imgsz=640,
    conf=CONFIDENCE,
    device="cpu",
    verbose=False
)[0]


# ============================================================
# CREATE BASE WIRE MASK
# ============================================================

wire_mask = detect_wire_mask(
    image
)


# ============================================================
# REMOVE COMPONENT REGIONS
# ============================================================
#
# Pixels inside detected component bounding boxes are removed
# from the wire mask.
#
# This prevents edges belonging to:
# - Arduino
# - Breadboard
# - ICs
# - sensors
# - motors
# - other detected components
#
# from being interpreted as jumper wires.
#
# ============================================================

component_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)


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

    # Clamp coordinates to image bounds
    x1 = max(
        0,
        min(width - 1, int(x1))
    )

    y1 = max(
        0,
        min(height - 1, int(y1))
    )

    x2 = max(
        0,
        min(width - 1, int(x2))
    )

    y2 = max(
        0,
        min(height - 1, int(y2))
    )

    class_name = model.names[
        class_id
    ]

    components.append(
        {
            "name": class_name,
            "confidence": confidence,
            "box": [
                x1,
                y1,
                x2,
                y2
            ]
        }
    )

    # Add a small safety margin around components.
    padding = 5

    px1 = max(
        0,
        x1 - padding
    )

    py1 = max(
        0,
        y1 - padding
    )

    px2 = min(
        width - 1,
        x2 + padding
    )

    py2 = min(
        height - 1,
        y2 + padding
    )

    cv2.rectangle(
        component_mask,
        (px1, py1),
        (px2, py2),
        255,
        -1
    )


# ============================================================
# REMOVE COMPONENT PIXELS
# ============================================================

wire_mask_clean = cv2.bitwise_and(
    wire_mask,
    cv2.bitwise_not(
        component_mask
    )
)


# ============================================================
# CLEAN THE MASK
# ============================================================

kernel = np.ones(
    (3, 3),
    np.uint8
)

wire_mask_clean = cv2.morphologyEx(
    wire_mask_clean,
    cv2.MORPH_OPEN,
    kernel
)

wire_mask_clean = cv2.morphologyEx(
    wire_mask_clean,
    cv2.MORPH_CLOSE,
    kernel
)


# ============================================================
# EDGE DETECTION
# ============================================================

edges = cv2.Canny(
    wire_mask_clean,
    50,
    150
)


# ============================================================
# HOUGH LINE DETECTION
# ============================================================

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=20,
    minLineLength=MIN_LINE_LENGTH,
    maxLineGap=MAX_LINE_GAP
)


# ============================================================
# EXTRACT SEGMENTS
# ============================================================

segments = []


if lines is not None:

    for line in lines:

        values = np.asarray(
            line
        ).reshape(-1)

        if values.size < 4:
            continue

        x1 = int(values[0])
        y1 = int(values[1])
        x2 = int(values[2])
        y2 = int(values[3])

        length = float(
            np.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )
        )

        if length < MIN_LINE_LENGTH:
            continue

        segments.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "length": length
            }
        )


# ============================================================
# DRAW DEBUG IMAGE
# ============================================================

debug_image = image.copy()


# Draw component boxes in green
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


# Draw candidate wire segments in magenta
for segment in segments:

    cv2.line(
        debug_image,
        (
            segment["x1"],
            segment["y1"]
        ),
        (
            segment["x2"],
            segment["y2"]
        ),
        (255, 0, 255),
        2
    )


# Save debug image
cv2.imwrite(
    OUTPUT_MASK_PATH,
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
    "SNAPLAB AI — COMPONENT-AWARE WIRE TEST"
)

print(
    "=" * 60
)

print(
    f"Input image: {IMAGE_PATH}"
)

print(
    f"Image size: "
    f"{width} x {height}"
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
    f"Original wire-mask pixels: "
    f"{int(np.count_nonzero(wire_mask))}"
)

print(
    f"Clean wire-mask pixels: "
    f"{int(np.count_nonzero(wire_mask_clean))}"
)

print()

print(
    f"Candidate wire segments: "
    f"{len(segments)}"
)

for index, segment in enumerate(
    segments[:20],
    start=1
):

    print(
        f"\nSegment {index}"
    )

    print(
        f"  Start: "
        f"({segment['x1']}, "
        f"{segment['y1']})"
    )

    print(
        f"  End: "
        f"({segment['x2']}, "
        f"{segment['y2']})"
    )

    print(
        f"  Length: "
        f"{segment['length']:.1f} px"
    )


if len(segments) > 20:

    print(
        f"\n... "
        f"{len(segments) - 20} more segments"
    )


print()

print(
    f"Debug image saved: "
    f"{OUTPUT_MASK_PATH}"
)

print()

print(
    "=" * 60
)

print(
    "COMPONENT-AWARE WIRE TEST: PASSED"
)

print(
    "=" * 60
)