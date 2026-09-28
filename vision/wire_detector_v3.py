import cv2
import numpy as np


# ============================================================
# SNAPLAB AI — WIRE DETECTOR V3
# ============================================================
#
# More conservative wire detector.
#
# V2 problem:
#   Bright pixels + Hough edges produced many false positives
#   from breadboard geometry, text and background.
#
# V3 strategy:
#   1. Detect only strongly colored pixels
#   2. Ignore bright/low-saturation regions
#   3. Remove detected component regions
#   4. Find connected colored regions
#   5. Keep regions that look elongated/thin
#
# This detects WIRE CANDIDATES.
# It does NOT prove electrical connectivity.
# ============================================================


def create_colored_wire_mask(
    image,
    min_saturation=100,
    min_value=60
):
    """
    Create a mask containing strongly colored pixels.

    White/gray breadboard/background regions are intentionally
    excluded.
    """

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(
        [
            0,
            min_saturation,
            min_value
        ],
        dtype=np.uint8
    )

    upper = np.array(
        [
            179,
            255,
            255
        ],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    return mask


def remove_component_regions(
    mask,
    component_boxes,
    padding=8
):
    """
    Remove detected component regions from the wire mask.

    Parameters
    ----------
    mask:
        Binary wire candidate mask.

    component_boxes:
        List of [x1, y1, x2, y2].

    padding:
        Extra exclusion margin around components.
    """

    clean_mask = mask.copy()

    height, width = clean_mask.shape[:2]

    for box in component_boxes:

        x1, y1, x2, y2 = box

        x1 = max(
            0,
            int(x1) - padding
        )

        y1 = max(
            0,
            int(y1) - padding
        )

        x2 = min(
            width - 1,
            int(x2) + padding
        )

        y2 = min(
            height - 1,
            int(y2) + padding
        )

        cv2.rectangle(
            clean_mask,
            (x1, y1),
            (x2, y2),
            0,
            -1
        )

    return clean_mask


def clean_wire_mask(
    mask
):
    """
    Morphological cleanup while preserving thin colored paths.
    """

    if mask is None:
        raise ValueError(
            "Mask is None."
        )

    # Small opening removes isolated colored noise.
    open_kernel = np.ones(
        (2, 2),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    # Small closing reconnects nearby wire pixels.
    close_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    return cleaned


def extract_wire_candidates(
    mask,
    min_area=20,
    min_aspect_ratio=2.0
):
    """
    Extract elongated connected colored regions.

    A wire candidate should generally be much longer than
    it is wide.

    Returns a list of candidate regions.
    """

    if mask is None:
        raise ValueError(
            "Mask is None."
        )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    candidates = []

    for label_id in range(
        1,
        num_labels
    ):

        x = int(
            stats[label_id, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[label_id, cv2.CC_STAT_TOP]
        )

        width = int(
            stats[label_id, cv2.CC_STAT_WIDTH]
        )

        height = int(
            stats[label_id, cv2.CC_STAT_HEIGHT]
        )

        area = int(
            stats[label_id, cv2.CC_STAT_AREA]
        )

        if area < min_area:
            continue

        long_side = max(
            width,
            height
        )

        short_side = max(
            1,
            min(width, height)
        )

        aspect_ratio = (
            long_side /
            short_side
        )

        if aspect_ratio < min_aspect_ratio:
            continue

        center_x = float(
            centroids[label_id][0]
        )

        center_y = float(
            centroids[label_id][1]
        )

        candidates.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "center_x": center_x,
                "center_y": center_y
            }
        )

    return candidates


def detect_wire_candidates(
    image,
    component_boxes=None,
    min_saturation=100,
    min_value=60,
    min_area=20,
    min_aspect_ratio=2.0
):
    """
    Complete V3 wire-candidate pipeline.

    Returns:
        raw_mask
        clean_mask
        candidates
    """

    raw_mask = create_colored_wire_mask(
        image,
        min_saturation=min_saturation,
        min_value=min_value
    )

    if component_boxes is None:
        component_boxes = []

    clean_mask = remove_component_regions(
        raw_mask,
        component_boxes
    )

    clean_mask = clean_wire_mask(
        clean_mask
    )

    candidates = extract_wire_candidates(
        clean_mask,
        min_area=min_area,
        min_aspect_ratio=min_aspect_ratio
    )

    return (
        raw_mask,
        clean_mask,
        candidates
    )


def draw_wire_candidates(
    image,
    candidates
):
    """
    Draw candidate wire regions on an image.
    """

    output = image.copy()

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        x = candidate["x"]
        y = candidate["y"]

        width = candidate["width"]
        height = candidate["height"]

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (255, 0, 255),
            2
        )

        label = (
            f"wire? {index} "
            f"{candidate['aspect_ratio']:.1f}"
        )

        cv2.putText(
            output,
            label,
            (x, max(15, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 255),
            1,
            cv2.LINE_AA
        )

    return output