from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import yaml
import random
import math

root = Path("datasets/electronic_components_3024_raw")
data = yaml.safe_load((root / "data.yaml").read_text())
names = data["names"]

img_dir = root / "train" / "images"
label_dir = root / "train" / "labels"

files = list(img_dir.glob("*"))
random.seed(42)
samples = random.sample(files, min(24, len(files)))

thumb_w, thumb_h = 320, 320
cols = 4
rows = math.ceil(len(samples) / cols)

sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
draw = ImageDraw.Draw(sheet)

for idx, img_path in enumerate(samples):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((thumb_w, thumb_h))

    xoff = (idx % cols) * thumb_w
    yoff = (idx // cols) * thumb_h

    sheet.paste(img, (xoff, yoff))

    label_path = label_dir / (img_path.stem + ".txt")

    if label_path.exists():
        lines = label_path.read_text().splitlines()

        for line in lines:
            p = line.split()
            if len(p) < 5:
                continue

            cls = int(p[0])
            vals = list(map(float, p[1:]))

            # Standard YOLO bbox
            if len(vals) == 4:
                xc, yc, bw, bh = vals
                iw, ih = img.size

                x1 = int((xc - bw / 2) * iw)
                y1 = int((yc - bh / 2) * ih)
                x2 = int((xc + bw / 2) * iw)
                y2 = int((yc + bh / 2) * ih)

                draw.rectangle(
                    (xoff+x1, yoff+y1, xoff+x2, yoff+y2),
                    outline="red",
                    width=3
                )

                draw.text(
                    (xoff+x1+2, yoff+y1+2),
                    names[cls],
                    fill="red"
                )

out = Path("runs/dataset_check/components3024_groundtruth.jpg")
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out, quality=95)

print("Created:", out)
print("Samples:", len(samples))
