from pathlib import Path
import shutil

src = Path("datasets/electronic_components_3024_raw")
dst = Path("datasets/snaplab_master")

class_map = {
    0: 7,    # BJT -> BJT-Transistor
    1: 16,   # Diode -> Diode
    2: 40,   # MOSFET -> MOSFET
    3: 61,   # OP_AMP -> OP-Amp
    4: 50,   # Resistor -> Resistor
    5: 62,   # cable -> Cable
    6: 63,   # capacitor -> Generic-Capacitor
    7: 64,   # variable_resistor -> Variable-Resistor
}

splits = {
    "train": "train",
    "valid": "val",
}

for source_split, master_split in splits.items():

    src_images = src / source_split / "images"
    src_labels = src / source_split / "labels"

    dst_images = dst / "images" / master_split
    dst_labels = dst / "labels" / master_split

    image_count = 0
    label_count = 0
    annotation_count = 0

    for image in src_images.iterdir():

        if not image.is_file():
            continue

        # Prevent filename collisions with other datasets.
        new_name = "ec3024_" + image.name

        shutil.copy2(
            image,
            dst_images / new_name
        )

        image_count += 1

        source_label = src_labels / (image.stem + ".txt")
        destination_label = dst_labels / (image.stem + ".txt")

        output_lines = []

        for line in source_label.read_text(errors="ignore").splitlines():

            parts = line.split()

            if len(parts) != 5:
                continue

            old_class = int(parts[0])

            if old_class not in class_map:
                continue

            new_class = class_map[old_class]

            output_lines.append(
                str(new_class) + " " + " ".join(parts[1:])
            )

            annotation_count += 1

        # Match the renamed image.
        destination_label = dst_labels / (new_name.rsplit(".", 1)[0] + ".txt")

        destination_label.write_text(
            "\n".join(output_lines) + ("\n" if output_lines else ""),
            encoding="utf-8"
        )

        label_count += 1

    print(
        f"{source_split:5s} -> {master_split:5s} | "
        f"images={image_count} labels={label_count} "
        f"annotations={annotation_count}"
    )

print("\n3024 dataset train/validation merge completed.")
print("3024 TEST SET was intentionally NOT merged.")
