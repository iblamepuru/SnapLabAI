from pathlib import Path
from PIL import Image
import yaml

root = Path("datasets/electronic_components_3024_raw")

data = yaml.safe_load((root / "data.yaml").read_text())
names = data["names"]

stats = {
    "images": 0,
    "labels": 0,
    "bad_labels": 0,
    "bad_boxes": 0,
    "tiny_boxes": 0,
    "empty_labels": 0,
}

tiny_examples = []

for split in ["train", "valid", "test"]:
    img_dir = root / split / "images"
    label_dir = root / split / "labels"

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        stats["images"] += 1

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue

        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            stats["empty_labels"] += 1
            continue

        stats["labels"] += 1
        lines = label_path.read_text(errors="ignore").splitlines()

        if not lines:
            stats["empty_labels"] += 1
            continue

        for line in lines:
            p = line.split()

            if len(p) != 5:
                stats["bad_labels"] += 1
                continue

            try:
                cls = int(p[0])
                xc, yc, bw, bh = map(float, p[1:])

                if cls < 0 or cls >= len(names):
                    stats["bad_labels"] += 1
                    continue

                if not all(0 <= v <= 1 for v in [xc, yc, bw, bh]):
                    stats["bad_boxes"] += 1
                    continue

                if bw <= 0 or bh <= 0:
                    stats["bad_boxes"] += 1
                    continue

                area = bw * bh

                if area < 0.0005:
                    stats["tiny_boxes"] += 1

                    if len(tiny_examples) < 10:
                        tiny_examples.append(
                            (split, img_path.name, names[cls], round(area, 6))
                        )

            except Exception:
                stats["bad_labels"] += 1

print("\n========== 3024 DATASET QUALITY AUDIT ==========")
for k, v in stats.items():
    print(f"{k:15s}: {v}")

print("\nTiny-box examples:")
for x in tiny_examples:
    print(x)

print("\nClasses:")
for i, name in enumerate(names):
    print(f"{i}: {name}")
