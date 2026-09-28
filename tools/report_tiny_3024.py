from pathlib import Path
from PIL import Image
import yaml

root = Path("datasets/electronic_components_3024_raw")
data = yaml.safe_load((root / "data.yaml").read_text())
names = data["names"]

print("\n========== TINY BOX REPORT ==========\n")

for split in ["train", "valid", "test"]:
    img_dir = root / split / "images"
    label_dir = root / split / "labels"

    for img_path in img_dir.iterdir():
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                iw, ih = im.size
        except:
            continue

        objects = []

        for line in label_path.read_text(errors="ignore").splitlines():
            p = line.split()
            if len(p) != 5:
                continue

            cls = int(p[0])
            xc, yc, bw, bh = map(float, p[1:])
            area = bw * bh

            objects.append((names[cls], area, bw, bh))

        tiny = [x for x in objects if x[1] < 0.0005]

        if tiny:
            print(f"{split.upper()} | {img_path.name}")
            print(f"  Resolution : {iw} x {ih}")
            print(f"  Objects    : {len(objects)}")

            for name, area, bw, bh in objects:
                marker = " <-- TINY" if area < 0.0005 else ""
                print(
                    f"    {name:18s} "
                    f"area={area:.6f} "
                    f"w={bw:.4f} h={bh:.4f}{marker}"
                )

            print()

