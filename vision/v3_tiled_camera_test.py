from ultralytics import YOLO
import cv2
import numpy as np

MODEL = r"runs\detect\runs\snaplab\phaseC_master_65class-2\weights\best.pt"

model = YOLO(MODEL)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

CONF = 0.20
IOU = 0.45
TILE_SCALE = 0.60
OVERLAP = 0.20

def tiled_detect(frame):
    h, w = frame.shape[:2]

    tile_w = int(w * TILE_SCALE)
    tile_h = int(h * TILE_SCALE)

    step_x = max(1, int(tile_w * (1 - OVERLAP)))
    step_y = max(1, int(tile_h * (1 - OVERLAP)))

    boxes = []
    scores = []
    classes = []

    xs = list(range(0, max(1, w - tile_w + 1), step_x))
    ys = list(range(0, max(1, h - tile_h + 1), step_y))

    if xs[-1] != w - tile_w:
        xs.append(w - tile_w)

    if ys[-1] != h - tile_h:
        ys.append(h - tile_h)

    for y1 in ys:
        for x1 in xs:
            x2 = x1 + tile_w
            y2 = y1 + tile_h

            tile = frame[y1:y2, x1:x2]

            result = model.predict(
                source=tile,
                imgsz=640,
                conf=CONF,
                verbose=False
            )[0]

            if result.boxes is None:
                continue

            for box, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy()
            ):
                bx1, by1, bx2, by2 = box

                boxes.append([
                    bx1 + x1,
                    by1 + y1,
                    bx2 + x1,
                    by2 + y1
                ])
                scores.append(float(conf))
                classes.append(int(cls))

    if not boxes:
        return []

    boxes_np = np.array(boxes, dtype=np.float32)

    # Class-wise NMS
    keep = []

    for cls_id in sorted(set(classes)):
        indices = [
            i for i, c in enumerate(classes)
            if c == cls_id
        ]

        cls_boxes = boxes_np[indices]
        cls_scores = np.array(
            [scores[i] for i in indices],
            dtype=np.float32
        )

        # Manual NMS
        order = cls_scores.argsort()[::-1]

        while len(order) > 0:
            current = order[0]
            keep.append(indices[current])

            if len(order) == 1:
                break

            current_box = cls_boxes[current]

            xx1 = np.maximum(
                current_box[0],
                cls_boxes[order[1:], 0]
            )
            yy1 = np.maximum(
                current_box[1],
                cls_boxes[order[1:], 1]
            )
            xx2 = np.minimum(
                current_box[2],
                cls_boxes[order[1:], 2]
            )
            yy2 = np.minimum(
                current_box[3],
                cls_boxes[order[1:], 3]
            )

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            intersection = inter_w * inter_h

            area_current = (
                (current_box[2] - current_box[0]) *
                (current_box[3] - current_box[1])
            )

            areas = (
                (cls_boxes[order[1:], 2] -
                 cls_boxes[order[1:], 0]) *
                (cls_boxes[order[1:], 3] -
                 cls_boxes[order[1:], 1])
            )

            union = area_current + areas - intersection
            iou = intersection / (union + 1e-6)

            order = order[1:][iou < IOU]

    detections = []

    for i in keep:
        detections.append({
            "box": boxes[i],
            "confidence": scores[i],
            "class_id": classes[i],
            "class_name": model.names[classes[i]]
        })

    return detections


print("=" * 60)
print("SNAPLAB V3 — TILED SMALL-OBJECT TEST")
print("=" * 60)
print("Model: 65-class YOLO11n")
print("Confidence:", CONF)
print("Tile scale:", TILE_SCALE)
print("Overlap:", OVERLAP)
print("Press Q to quit")
print("=" * 60)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame read failed")
        break

    detections = tiled_detect(frame)

    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        conf = det["confidence"]
        name = det["class_name"]

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            2
        )

        label = f"{name} {conf:.2f}"

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2
        )

    cv2.putText(
        annotated,
        f"Tiled detections: {len(detections)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "SnapLab V3 - Tiled Detection",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Tiled V3 camera test completed.")
