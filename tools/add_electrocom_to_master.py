from pathlib import Path
import shutil

src = Path("datasets/electrocom61_raw")
dst = Path("datasets/snaplab_master")

splits = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

for source_split, master_split in splits.items():

    src_images = src / source_split / "images"
    src_labels = src / source_split / "labels"

    dst_images = dst / "images" / master_split
    dst_labels = dst / "labels" / master_split

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0

    for image in src_images.iterdir():
        if image.is_file():
            shutil.copy2(image, dst_images / image.name)
            image_count += 1

    for label in src_labels.glob("*.txt"):
        shutil.copy2(label, dst_labels / label.name)
        label_count += 1

    print(
        f"{source_split:5s} -> {master_split:5s} | "
        f"images={image_count} labels={label_count}"
    )

print("\nElectroCom-61 successfully added to master dataset.")
