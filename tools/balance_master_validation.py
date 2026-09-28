from pathlib import Path
from collections import Counter, defaultdict
import shutil

root = Path("datasets/snaplab_master")

train_img = root / "images" / "train"
train_lbl = root / "labels" / "train"
val_img = root / "images" / "val"
val_lbl = root / "labels" / "val"

TARGET_MIN = 10

# ---------------------------------------------------------
# Find current validation image coverage per class
# ---------------------------------------------------------

val_class_images = defaultdict(set)

for label_file in val_lbl.glob("*.txt"):
    classes = set()

    for line in label_file.read_text(errors="ignore").splitlines():
        if line.strip():
            classes.add(int(line.split()[0]))

    for cls in classes:
        val_class_images[cls].add(label_file.stem)

# ---------------------------------------------------------
# Find training candidates for underrepresented classes
# ---------------------------------------------------------

train_class_images = defaultdict(list)

for label_file in train_lbl.glob("*.txt"):

    classes = set()

    for line in label_file.read_text(errors="ignore").splitlines():
        if line.strip():
            classes.add(int(line.split()[0]))

    for cls in classes:
        train_class_images[cls].append(label_file)

# ---------------------------------------------------------
# Determine classes needing additional validation images
# ---------------------------------------------------------

all_classes = set(train_class_images) | set(val_class_images)

needs = {}

for cls in sorted(all_classes):

    current = len(val_class_images[cls])

    if current < TARGET_MIN:
        needs[cls] = TARGET_MIN - current

print("========== VALIDATION BALANCING ==========")
print("Target minimum validation images per class:", TARGET_MIN)
print("Classes requiring additional validation data:", len(needs))
print()

# ---------------------------------------------------------
# Select candidates
# Prefer images containing rare classes.
# An image selected once can satisfy multiple classes.
# ---------------------------------------------------------

selected = set()
selection_reason = defaultdict(list)

while True:

    remaining = {
        cls: need
        for cls, need in needs.items()
        if need > 0
    }

    if not remaining:
        break

    best_file = None
    best_gain = 0

    for label_file in train_lbl.glob("*.txt"):

        if label_file.stem in selected:
            continue

        classes = set()

        for line in label_file.read_text(errors="ignore").splitlines():
            if line.strip():
                classes.add(int(line.split()[0]))

        gain = sum(
            1
            for cls in remaining
            if cls in classes
        )

        if gain > best_gain:
            best_gain = gain
            best_file = label_file

    if best_file is None or best_gain == 0:
        break

    selected.add(best_file.stem)

    classes = set()

    for line in best_file.read_text(errors="ignore").splitlines():
        if line.strip():
            classes.add(int(line.split()[0]))

    for cls in remaining:
        if cls in classes:
            needs[cls] -= 1
            selection_reason[best_file.stem].append(cls)

# ---------------------------------------------------------
# Move selected images + labels
# ---------------------------------------------------------

moved = 0
missing_images = 0

for stem in sorted(selected):

    label_file = train_lbl / f"{stem}.txt"

    # Locate corresponding image
    image_file = None

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = train_img / f"{stem}{ext}"
        if candidate.exists():
            image_file = candidate
            break

    if image_file is None:
        missing_images += 1
        continue

    shutil.move(str(image_file), str(val_img / image_file.name))
    shutil.move(str(label_file), str(val_lbl / label_file.name))

    moved += 1

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("Images selected :", len(selected))
print("Images moved    :", moved)
print("Missing images  :", missing_images)
print()

if needs:
    print("Classes still below target:")
    for cls, remaining in sorted(needs.items()):
        if remaining > 0:
            print(f"Class {cls}: still needs {remaining}")

print()
print("========== BALANCING COMPLETE ==========")
