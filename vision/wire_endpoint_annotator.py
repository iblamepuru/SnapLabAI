import cv2
import json
from pathlib import Path

IMAGE_PATH = Path(r"datasets\Jumper_Wire.jpg")
OUTPUT_PATH = Path(r"results\jumper_wire_endpoints.json")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

points = []
wires = []
preview = image.copy()

def redraw():
    global preview
    preview = image.copy()

    for wire in wires:
        a = tuple(wire["start_point"])
        b = tuple(wire["end_point"])
        cv2.line(preview, a, b, (255, 0, 255), 2)
        cv2.circle(preview, a, 6, (0, 255, 0), -1)
        cv2.circle(preview, b, 6, (0, 0, 255), -1)
        cv2.putText(
            preview,
            f'{wire["wire_id"]}: {wire["color"]}',
            (a[0], max(20, a[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1
        )

    for i, point in enumerate(points):
        cv2.circle(preview, point, 7, (0, 255, 255), -1)
        cv2.putText(
            preview,
            f"P{i + 1}",
            (point[0] + 8, point[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        redraw()

cv2.namedWindow("Wire Endpoint Annotator", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Wire Endpoint Annotator", mouse_callback)

print("Click the two physical endpoints of one wire.")
print("Then enter the wire details in PowerShell.")
print("Press U to undo the last point.")
print("Press S to save all annotations.")
print("Press Q to quit.")

while True:
    cv2.imshow("Wire Endpoint Annotator", preview)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("u") and points:
        points.pop()
        redraw()

    elif key == ord("s"):
        if len(points) % 2 != 0:
            print("You have an unmatched endpoint. Click its second endpoint first.")
            continue

        if len(points) == 0:
            print("No endpoints annotated yet.")
            continue

        cv2.destroyAllWindows()

        while True:
            if not points:
                break

            start_point = points.pop(0)
            end_point = points.pop(0)

            wire_id = len(wires) + 1
            print(f"\nWire {wire_id}")

            color = input("Wire color: ").strip()
            start_terminal = input("Start terminal label: ").strip()
            end_terminal = input("End terminal label: ").strip()

            wires.append({
                "wire_id": wire_id,
                "color": color,
                "start_point": start_point,
                "end_point": end_point,
                "start_terminal": start_terminal,
                "end_terminal": end_terminal
            })

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps({
                "image": str(IMAGE_PATH),
                "image_width": image.shape[1],
                "image_height": image.shape[0],
                "wires": wires
            }, indent=2),
            encoding="utf-8"
        )

        print(f"\nSaved {len(wires)} wires to {OUTPUT_PATH.resolve()}")
        break

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
