import cv2

def crop_detections(image, result, padding=10, scale=2.0):
    h, w = image.shape[:2]
    crops = []

    for box, cls, conf in zip(
        result.boxes.xyxy,
        result.boxes.cls,
        result.boxes.conf
    ):
        x1, y1, x2, y2 = [int(v) for v in box]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        if scale != 1.0:
            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        crops.append({
            "image": crop,
            "class_id": int(cls),
            "confidence": float(conf),
            "box": [x1, y1, x2, y2]
        })

    return crops
