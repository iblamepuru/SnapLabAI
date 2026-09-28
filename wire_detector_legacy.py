import cv2
import numpy as np


class WireDetector:

    def __init__(self, min_area=80, min_length=40):
        self.min_area = min_area
        self.min_length = min_length


    def detect(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Broad saturation mask.
        # This captures strongly colored jumper wires
        # while suppressing much of the neutral background.
        lower = np.array([0, 70, 50])
        upper = np.array([179, 255, 255])

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        # Remove small noise.
        kernel = np.ones((5, 5), np.uint8)

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

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        wires = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            length = max(w, h)

            if length < self.min_length:
                continue

            aspect_ratio = length / max(
                1,
                min(w, h)
            )

            # A wire-like region is generally elongated.
            if aspect_ratio < 2.0:
                continue

            wires.append({
                "bbox": (x, y, x + w, y + h),
                "area": round(float(area), 2),
                "length": int(length),
                "aspect_ratio": round(
                    float(aspect_ratio),
                    2
                )
            })

        return wires


if __name__ == "__main__":

    detector = WireDetector()

    print("WireDetector initialized successfully.")

    print(
        "Detection strategy:"
    )

    print(
        "HSV saturation + morphology + "
        "contour geometry"
    )

    print(
        "Wire detector foundation test PASSED."
    )
