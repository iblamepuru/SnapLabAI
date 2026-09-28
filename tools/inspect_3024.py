from pathlib import Path
from collections import Counter
import yaml

root = Path("datasets/electronic_components_3024_raw")

data = yaml.safe_load((root / "data.yaml").read_text())
names = data["names"]

total_counts = Counter()
split_counts = {}

for split in ["train", "valid", "test"]:
    label_dir = root / split / "labels"
    counts = Counter()

    for f in label_dir.glob("*.txt"):
        for line in f.read_text(errors="ignore").splitlines():
            parts = line.split()

            if parts and parts[0].isdigit():
                class_id = int(parts[0])
                counts[class_id] += 1

    split_counts[split] = counts
    total_counts.update(counts)

print(f"TOTAL ANNOTATIONS: {sum(total_counts.values())}")
print()

for i, name in enumerate(names):
    print(
        f"{i:2d} {name:18s} "
        f"total={total_counts[i]:5d} "
        f"train={split_counts['train'][i]:5d} "
        f"valid={split_counts['valid'][i]:5d} "
        f"test={split_counts['test'][i]:5d}"
    )
