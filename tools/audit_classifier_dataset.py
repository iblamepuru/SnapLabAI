from pathlib import Path

root = Path("datasets/component_classifier")

print("CLASS DISTRIBUTION")
print("=" * 60)

for split in ["train", "val", "test"]:
    split_dir = root / split
    print()
    print(split.upper())
    print("-" * 60)

    total = 0

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        count = len(list(class_dir.glob("*.jpg")))
        total += count
        print(f"{class_dir.name:35} {count}")

    print("-" * 60)
    print(f"{'TOTAL':35} {total}")