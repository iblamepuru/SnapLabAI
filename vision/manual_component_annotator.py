import cv2
import json
from pathlib import Path

IMAGE_PATH = Path(r"datasets\Jumper_Wire.jpg")
OUTPUT_PATH = Path(r"results\jumper_wire_manual_components.json")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

components = []
drawing = False
start_point = None
preview = image.copy()

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, preview

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        preview = image.copy()
        cv2.rectangle(preview, start_point, (x, y), (0, 255, 0), 2)

        for item in components:
            x1, y1, x2, y2 = item["bbox"]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(preview, f'{item["track_id"]}: {item["class_name"]}',
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 1)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y

        x1, x2 = sorted([max(0, min(x1, image.shape[1] - 1)),
                         max(0, min(x2, image.shape[1] - 1))])
        y1, y2 = sorted([max(0, min(y1, image.shape[0] - 1)),
                         max(0, min(y2, image.shape[0] - 1))])

        if x2 - x1 >= 5 and y2 - y1 >= 5:
            name = input("Component name (e.g. Breadboard): ").strip()

            if name:
                components.append({
                    "track_id": len(components) + 1,
                    "class_name": name,
                    "confidence": 1.0,
                    "bbox": [x1, y1, x2, y2]
                })

        preview = image.copy()

        for item in components:
            a, b, c, d = item["bbox"]
            cv2.rectangle(preview, (a, b), (c, d), (255, 0, 255), 2)
            cv2.putText(preview, f'{item["track_id"]}: {item["class_name"]}',
                        (a, max(20, b - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 1)

cv2.namedWindow("Manual Component Annotation", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Manual Component Annotation", mouse_callback)

print("Drag a box around each visible component.")
print("Press U to undo the last box.")
print("Press S to save.")
print("Press Q to quit without saving.")

while True:
    cv2.imshow("Manual Component Annotation", preview)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("u") and components:
        components.pop()
        preview = image.copy()

        for item in components:
            x1, y1, x2, y2 = item["bbox"]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(preview, f'{item["track_id"]}: {item["class_name"]}',
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 1)

    elif key == ord("s"):
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps({
            "image": str(IMAGE_PATH),
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "components": components
        }, indent=2))

        print(f"Saved {len(components)} components to {OUTPUT_PATH}")
        break

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
