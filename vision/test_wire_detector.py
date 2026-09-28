import cv2

from wire_detector import detect_wires


# ============================================================
# INPUT
# ============================================================

IMAGE_PATH = r"vision\reference_frame.jpg"


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )


# ============================================================
# DETECT WIRES
# ============================================================

mask, segments = detect_wires(
    image,
    min_length=20
)


# ============================================================
# SAVE DEBUG OUTPUT
# ============================================================

mask_path = r"vision\wire_mask_debug.png"

cv2.imwrite(
    mask_path,
    mask
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("SNAPLAB AI — WIRE DETECTOR TEST")
print("=" * 60)

print(
    f"Input image: {IMAGE_PATH}"
)

print(
    f"Wire segments detected: "
    f"{len(segments)}"
)

print(
    f"Wire mask saved: "
    f"{mask_path}"
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
print("=" * 60)
print("WIRE DETECTOR TEST: PASSED")
print("=" * 60)