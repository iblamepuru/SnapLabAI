from pathlib import Path
import yaml

root = Path("datasets/snaplab_master")
data = yaml.safe_load((root / "data.yaml").read_text())
num_classes = data["nc"]

total = 0
bad = []

for split in ["train", "val", "test"]:
    label_dir = root / "labels" / split

    for f in label_dir.glob("*.txt"):
        for line_no, line in enumerate(
            f.read_text(errors="ignore").splitlines(), 1
        ):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                bad.append((str(f), line_no, "field_count", line))
                continue

            try:
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                bad.append((str(f), line_no, "non_numeric", line))
                continue

            total += 1

            if not (0 <= cls < num_classes):
                bad.append((str(f), line_no, "class_id", line))
                continue

            if not all(0 <= v <= 1 for v in [xc, yc, w, h]):
                bad.append((str(f), line_no, "coordinate_range", line))
                continue

            if w <= 0 or h <= 0:
                bad.append((str(f), line_no, "zero_size", line))
                continue

            if xc - w/2 < 0 or xc + w/2 > 1:
                bad.append((str(f), line_no, "x_out_of_bounds", line))
                continue

            if yc - h/2 < 0 or yc + h/2 > 1:
                bad.append((str(f), line_no, "y_out_of_bounds", line))
                continue

print("\n========== MASTER DATASET VALIDATION ==========")
print("Classes:", num_classes)
print("Valid annotation lines:", total)
print("Invalid annotation lines:", len(bad))

if bad:
    print("\nFirst 10 problems:")
    for item in bad[:10]:
        print(item)
else:
    print("\nALL LABELS PASSED.")
