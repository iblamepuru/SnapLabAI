from pathlib import Path
from collections import Counter
import yaml

root = Path("datasets/snaplab_master")
names = yaml.safe_load((root / "data.yaml").read_text())["names"]

print("========== IMAGE-LEVEL CLASS DISTRIBUTION ==========")

for split in ["train", "val", "test"]:

    image_counts = Counter()

    label_dir = root / "labels" / split

    for label_file in label_dir.glob("*.txt"):

        classes_in_image = set()

        for line in label_file.read_text(errors="ignore").splitlines():

            if line.strip():
                cls = int(line.split()[0])
                classes_in_image.add(cls)

        for cls in classes_in_image:
            image_counts[cls] += 1

    print()
    print(f"========== {split.upper()} ==========")

    for i, name in enumerate(names):
        print(f"{i:2d} {name:30s} {image_counts[i]:6d}")

    print("Images:", len(list(label_dir.glob("*.txt"))))

print()
print("========== REPORT COMPLETE ==========")
