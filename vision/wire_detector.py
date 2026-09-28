import cv2
import numpy as np


def detect_wire_mask(image):
    if image is None:
        raise ValueError("Input image is None.")

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    colored = cv2.inRange(
        hsv,
        np.array([0, 70, 50], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8)
    )

    brightness = hsv[:, :, 2]

    bright = cv2.inRange(
        brightness,
        150,
        255
    )

    mask = cv2.bitwise_or(
        colored,
        bright
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def extract_wire_segments(mask, min_length=20):
    if mask is None:
        raise ValueError("Wire mask is None.")

    edges = cv2.Canny(
        mask,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=min_length,
        maxLineGap=15
    )

    segments = []

    if lines is None:
        return segments

    for line in lines:
        values = np.asarray(line).reshape(-1)

        if values.size < 4:
            continue

        x1 = int(values[0])
        y1 = int(values[1])
        x2 = int(values[2])
        y2 = int(values[3])

        dx = x2 - x1
        dy = y2 - y1

        length = float(
            np.sqrt(
                dx * dx +
                dy * dy
            )
        )

        if length < min_length:
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

    return segments


def detect_wires(image, min_length=20):
    mask = detect_wire_mask(image)

    segments = extract_wire_segments(
        mask,
        min_length=min_length
    )

    return mask, segments


def draw_wire_segments(image, segments):
    if image is None:
        raise ValueError("Input image is None.")

    output = image.copy()

    for segment in segments:
        x1 = int(segment["x1"])
        y1 = int(segment["y1"])
        x2 = int(segment["x2"])
        y2 = int(segment["y2"])

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            2
        )

    return output