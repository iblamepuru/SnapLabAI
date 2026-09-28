from pathlib import Path

root = Path("datasets/component_classifier")

for split in ["train", "val", "test"]:
    split_dir = root / split

    print()
    print("=" * 60)
    print(f"{split.upper()} CLASS DISTRIBUTION")
    print("=" * 60)

    total = 0

    for cls in sorted(split_dir.iterdir()):
        if cls.is_dir():
            count = (
                len(list(cls.rglob("*.jpg"))) +
                len(list(cls.rglob("*.jpeg"))) +
                len(list(cls.rglob("*.png")))
            )

            print(f"{cls.name:35} {count}")
            total += count

    print("-" * 60)
    print(f"{'TOTAL':35} {total}")