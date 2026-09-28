from pathlib import Path
from collections import Counter
import yaml

root = Path("datasets/snaplab_master")
names = yaml.safe_load((root / "data.yaml").read_text())["names"]

print("========== MASTER SPLIT DISTRIBUTION ==========")

for split in ["train", "val", "test"]:

    counter = Counter()

    for label_file in (root / "labels" / split).glob("*.txt"):
        for line in label_file.read_text(errors="ignore").splitlines():
            if line.strip():
                counter[int(line.split()[0])] += 1

    print()
    print(f"========== {split.upper()} ==========")

    for i, name in enumerate(names):
        print(f"{i:2d} {name:30s} {counter[i]:6d}")

    print("Total annotations:", sum(counter.values()))

print()
print("========== REPORT COMPLETE ==========")
