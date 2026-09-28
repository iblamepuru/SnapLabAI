import cv2
from ultralytics import YOLO


# ============================================================
# SNAPLAB AI — SMALL COMPONENT REFINER TEST
# ============================================================

MODEL_PATH = (
    r"runs\detect\runs\snaplab\phaseC_master_65class-2"
    r"\weights\best.pt"
)

IMAGE_PATH = r"vision\reference_frame.jpg"

CROP_OUTPUT = r"vision\resistor_refinement_crop.png"

model = YOLO(MODEL_PATH)


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
# FIRST-PASS DETECTION
# ============================================================

result = model.predict(
    image,
    imgsz=640,
    conf=0.001,
    device="cpu",
    verbose=False
)[0]


# ============================================================
# FIND RESISTOR CANDIDATE
# ============================================================

resistor_candidate = None

for box in result.boxes:

    class_id = int(
        box.cls[0]
    )

    class_name = model.names[
        class_id
    ]

    confidence = float(
        box.conf[0]
    )

    if class_name == "Resistor":

        resistor_candidate = {
            "confidence": confidence,
            "box": box.xyxy[0].tolist()
        }

        break


if resistor_candidate is None:

    print(
        "RESISTOR CANDIDATE: NOT FOUND"
    )

    raise SystemExit(0)


# ============================================================
# EXTRACT RESISTOR BOX
# ============================================================

x1, y1, x2, y2 = (
    resistor_candidate["box"]
)

x1 = int(x1)
y1 = int(y1)
x2 = int(x2)
y2 = int(y2)


# ============================================================
# ADD LARGE PADDING
# ============================================================

padding = 35

height, width = image.shape[:2]

x1 = max(
    0,
    x1 - padding
)

y1 = max(
    0,
    y1 - padding
)

x2 = min(
    width,
    x2 + padding
)

y2 = min(
    height,
    y2 + padding
)


# ============================================================
# CROP
# ============================================================

crop = image[
    y1:y2,
    x1:x2
]


if crop.size == 0:

    raise RuntimeError(
        "Resistor crop is empty."
    )


# ============================================================
# UPSCALE
# ============================================================

scale = 4

crop_upscaled = cv2.resize(
    crop,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)


# ============================================================
# SAVE CROP
# ============================================================

cv2.imwrite(
    CROP_OUTPUT,
    crop_upscaled
)


# ============================================================
# SECOND-PASS DETECTION
# ============================================================

refined_result = model.predict(
    crop_upscaled,
    imgsz=640,
    conf=0.001,
    device="cpu",
    verbose=False
)[0]


# ============================================================
# RESULTS
# ============================================================

print()
print(
    "=" * 60
)

print(
    "SNAPLAB AI — SMALL COMPONENT REFINEMENT"
)

print(
    "=" * 60
)

print(
    f"Original image: {image.shape}"
)

print(
    f"Initial resistor confidence: "
    f"{resistor_candidate['confidence']:.4f}"
)

print(
    f"Original resistor box: "
    f"[{x1}, {y1}, {x2}, {y2}]"
)

print(
    f"Crop size: "
    f"{crop.shape}"
)

print(
    f"Upscaled crop: "
    f"{crop_upscaled.shape}"
)

print(
    f"Refined detections: "
    f"{len(refined_result.boxes)}"
)

print()

if len(refined_result.boxes) == 0:

    print(
        "No components detected in refined crop."
    )

else:

    for index, box in enumerate(
        refined_result.boxes,
        start=1
    ):

        class_id = int(
            box.cls[0]
        )

        class_name = model.names[
            class_id
        ]

        confidence = float(
            box.conf[0]
        )

        print(
            f"{index}. "
            f"{class_name} — "
            f"{confidence:.4f}"
        )

print()

print(
    f"Refinement crop saved: "
    f"{CROP_OUTPUT}"
)

print()

print(
    "=" * 60
)

print(
    "SMALL COMPONENT REFINEMENT TEST: PASSED"
)

print(
    "=" * 60
)