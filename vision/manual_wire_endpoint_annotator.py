import cv2
import json
from pathlib import Path

IMAGE_PATH = Path(r"datasets\Jumper_Wire.jpg")
OUTPUT_PATH = Path(r"results\jumper_wire_endpoint_pairs.json")
WIRES = [
    ("Wire 01", "red"),
    ("Wire 02", "black"),
    ("Wire 03", "blue"),
    ("Wire 04", "teal"),
    ("Wire 05", "yellow"),
    ("Wire 06", "gray"),
    ("Wire 07", "white"),
    ("Wire 08", "brown"),
    ("Wire 09", "orange"),
    ("Wire 10", "purple"),
]

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]
display_width = 800
display_height = 800
scale_x = width / display_width
scale_y = height / display_height

wire_index = 0
points = []
saved_wires = []

def render():
    canvas = image.copy()
    for i, wire in enumerate(saved_wires):
        for label, point in [("S", wire["start_point"]), ("E", wire["end_point"])]:
            x, y = point
            cv2.circle(canvas, (x, y), 7, (0, 255, 255), -1)
            cv2.putText(canvas, f'{wire["wire_id"]}{label}', (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    for i, point in enumerate(points):
        x, y = point
        cv2.circle(canvas, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(canvas, "START" if i == 0 else "END",
                    (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    name, color = WIRES[wire_index]
    status = f"{name} ({color}) | Click START, then END | N=next U=undo S=save Q=quit"
    cv2.putText(canvas, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2)
    return cv2.resize(canvas, (display_width, display_height))

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        original_x = int(round(x * scale_x))
        original_y = int(round(y * scale_y))
        points.append([original_x, original_y])
        print("START" if len(points) == 1 else "END", points[-1])

cv2.namedWindow("Wire Endpoint Annotation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Wire Endpoint Annotation", display_width, display_height)
cv2.setMouseCallback("Wire Endpoint Annotation", mouse_callback)

print("Click each wire's two physical insertion points.")
print("N = accept pair and continue | U = undo latest point | S = save | Q = quit")

while True:
    cv2.imshow("Wire Endpoint Annotation", render())
    key = cv2.waitKey(30) & 0xFF

    if key in (ord("n"), ord("N")):
        if len(points) != 2:
            print("Click both endpoints first.")
            continue
        name, color = WIRES[wire_index]
        saved_wires.append({
            "wire_id": wire_index + 1,
            "name": name,
            "color": color,
            "start_point": points[0],
            "end_point": points[1],
            "start_terminal": "upper_breadboard",
            "end_terminal": "lower_breadboard"
        })
        print(f"Accepted {name}: {points[0]} -> {points[1]}")
        points = []
        wire_index += 1
        if wire_index == len(WIRES):
            print("All 10 wire pairs annotated. Press S to save.")
            wire_index = len(WIRES) - 1

    elif key in (ord("u"), ord("U")):
        if points:
            points.pop()
            print("Undid last point.")

    elif key in (ord("s"), ord("S")):
        if len(saved_wires) != len(WIRES):
            print(f"Not saved: {len(saved_wires)} of {len(WIRES)} wires accepted.")
            continue
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image": str(IMAGE_PATH).replace("\\", "/"),
            "image_width": width,
            "image_height": height,
            "annotation_type": "manual_original_image_endpoint_pairs",
            "wires": saved_wires
        }
        OUTPUT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("Saved:", OUTPUT_PATH.resolve())
        break

    elif key in (ord("q"), ord("Q")):
        print("Quit without saving.")
        break

cv2.destroyAllWindows()
