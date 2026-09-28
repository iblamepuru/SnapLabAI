from pathlib import Path
import cv2
import math
import numpy as np

root = Path("datasets/component_classifier/train")
output = Path("datasets/component_classifier_preview")
output.mkdir(parents=True, exist_ok=True)

classes = sorted([p for p in root.iterdir() if p.is_dir()])

for class_dir in classes:
    images = sorted(class_dir.glob("*.jpg"))[:6]

    if not images:
        continue

    tiles = []

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image = cv2.resize(image, (160, 160))
        cv2.putText(
            image,
            class_dir.name[:20],
            (5, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        tiles.append(image)

    if not tiles:
        continue

    cols = 3
    rows = math.ceil(len(tiles) / cols)

    sheet = 255 * np.ones(
        (rows * 160, cols * 160, 3),
        dtype="uint8"
    )

    for i, tile in enumerate(tiles):
        y = (i // cols) * 160
        x = (i % cols) * 160
        sheet[y:y + 160, x:x + 160] = tile

    cv2.imwrite(
        str(output / f"{class_dir.name}.jpg"),
        sheet
    )

print("CLASSIFIER PREVIEW CREATED")
print("Output:", output)
print("Classes:", len(classes))