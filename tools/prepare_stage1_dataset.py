from pathlib import Path
import shutil

# ============================================================
# SnapLab-AI Stage 1 Dataset Preparation
# Roboflow Arduino segmentation -> SnapLab YOLO detection
# ============================================================

SOURCE = Path("datasets/roboflow_arduino")
TARGET = Path("datasets/electronics")

# ------------------------------------------------------------
# Roboflow class ID -> SnapLab fixed class ID
# ------------------------------------------------------------
CLASS_MAP = {
    0: 0,   # Arduino
    1: 1,   # Breadboard
    2: 4,   # Button -> Push_Button
    3: 7,   # Capacitor
    5: 5,   # Jumper-Wires -> Jumper_Wire
    6: 2,   # LED
    8: 3,   # Resistor
    9: 10,  # Transistor
}

# Roboflow uses "valid", while SnapLab uses "val"
SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ============================================================
# Convert segmentation polygon -> YOLO bounding box
# ============================================================

def polygon_to_bbox(parts):
    """
    Convert a YOLO segmentation annotation:

        class x1 y1 x2 y2 ... xn yn

    into YOLO detection format:

        class x_center y_center width height
    """

    class_id = int(parts[0])
    coordinates = list(map(float, parts[1:]))

    # Need at least 3 points -> 6 coordinates
    if len(coordinates) < 6:
        return None

    # Polygon must contain x,y pairs
    if len(coordinates) % 2 != 0:
        return None

    x_values = coordinates[0::2]
    y_values = coordinates[1::2]

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    width = x_max - x_min
    height = y_max - y_min

    # Reject degenerate boxes
    if width <= 0 or height <= 0:
        return None

    return (
        class_id,
        x_center,
        y_center,
        width,
        height
    )


# ============================================================
# Process one dataset split
# ============================================================

def prepare_split(source_split, target_split):

    source_images = SOURCE / source_split / "images"
    source_labels = SOURCE / source_split / "labels"

    target_images = TARGET / "images" / target_split
    target_labels = TARGET / "labels" / target_split

    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    unsupported_only = 0
    converted_annotations = 0
    bbox_annotations = 0

    for image_path in source_images.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"[WARNING] Missing label: {image_path.name}")
            skipped += 1
            continue

        output_lines = []

        with label_path.open("r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                # ------------------------------------------------
                # Basic validation
                # ------------------------------------------------

                try:
                    class_id = int(parts[0])
                except ValueError:
                    print(
                        f"[WARNING] Invalid class ID: "
                        f"{label_path.name}"
                    )
                    continue

                # ------------------------------------------------
                # Skip classes not included in Stage 1
                # ------------------------------------------------

                if class_id not in CLASS_MAP:
                    continue

                # ------------------------------------------------
                # Case 1: Standard YOLO bounding box
                #
                # class x_center y_center width height
                # ------------------------------------------------

                if len(parts) == 5:

                    try:
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError:
                        print(
                            f"[WARNING] Invalid bbox values: "
                            f"{label_path.name}"
                        )
                        continue

                    new_class_id = CLASS_MAP[class_id]

                    output_lines.append(
                        f"{new_class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}"
                    )

                    bbox_annotations += 1

                # ------------------------------------------------
                # Case 2: YOLO segmentation polygon
                #
                # class x1 y1 x2 y2 ... xn yn
                # ------------------------------------------------

                elif len(parts) >= 7:

                    bbox = polygon_to_bbox(parts)

                    if bbox is None:
                        print(
                            f"[WARNING] Invalid polygon: "
                            f"{label_path.name}"
                        )
                        continue

                    (
                        original_class_id,
                        x_center,
                        y_center,
                        width,
                        height
                    ) = bbox

                    new_class_id = CLASS_MAP[original_class_id]

                    output_lines.append(
                        f"{new_class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}"
                    )

                    converted_annotations += 1

                # ------------------------------------------------
                # Anything else is invalid
                # ------------------------------------------------

                else:

                    print(
                        f"[WARNING] Invalid annotation format: "
                        f"{label_path.name}"
                    )
                    continue

        # --------------------------------------------------------
        # If image contains only unsupported classes,
        # don't copy it into the Stage 1 detector dataset.
        # --------------------------------------------------------

        if not output_lines:
            unsupported_only += 1
            continue

        destination_image = target_images / image_path.name
        destination_label = target_labels / f"{image_path.stem}.txt"

        shutil.copy2(
            image_path,
            destination_image
        )

        with destination_label.open(
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(output_lines)
                + "\n"
            )

        copied += 1

    return (
        copied,
        skipped,
        unsupported_only,
        converted_annotations,
        bbox_annotations
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("SnapLab-AI Stage 1 Dataset Preparation")
    print("=" * 60)

    print("\nSource:")
    print(SOURCE.resolve())

    print("\nTarget:")
    print(TARGET.resolve())

    total_copied = 0
    total_converted = 0
    total_bbox = 0

    for source_split, target_split in SPLITS.items():

        print(
            f"\nProcessing "
            f"{source_split} -> {target_split}"
        )

        (
            copied,
            skipped,
            unsupported,
            converted,
            bbox
        ) = prepare_split(
            source_split,
            target_split
        )

        total_copied += copied
        total_converted += converted
        total_bbox += bbox

        print(f"  Copied images:        {copied}")
        print(f"  Missing/skipped:      {skipped}")
        print(f"  Unsupported-only:     {unsupported}")
        print(f"  Polygon -> bbox:      {converted}")
        print(f"  Existing bbox:        {bbox}")

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Total images copied:    {total_copied}")
    print(f"Polygon conversions:    {total_converted}")
    print(f"Existing bbox labels:   {total_bbox}")

    print("\nStage 1 taxonomy:")
    print("  0  Arduino")
    print("  1  Breadboard")
    print("  2  LED")
    print("  3  Resistor")
    print("  4  Push_Button")
    print("  5  Jumper_Wire")
    print("  7  Capacitor")
    print(" 10  Transistor")


if __name__ == "__main__":
    main()