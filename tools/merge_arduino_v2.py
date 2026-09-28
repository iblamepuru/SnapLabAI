from pathlib import Path
import shutil

SRC = Path("datasets/roboflow_arduino")
DST = Path("datasets/snaplab_v2")

CLASS_MAP = {
    0: 6,    # Arduino -> Arduino-Uno
    1: 9,    # Breadboard -> Breadboard
    2: 46,   # Button -> Push-Switch
    3: 13,   # Capacitor -> Capacitor-10mf
    6: 37,   # LED -> LED-Light
    8: 50,   # Resistor -> Resistor
    9: 7,    # Transistor -> BJT-Transistor
}

SOURCE_SPLITS = {
    "train": "train",
    "valid": "val",
}

for src_split, dst_split in SOURCE_SPLITS.items():
    src_images = SRC / src_split / "images"
    src_labels = SRC / src_split / "labels"

    dst_images = DST / "images" / dst_split
    dst_labels = DST / "labels" / dst_split

    copied = 0
    converted = 0
    skipped = 0

    for label_path in src_labels.glob("*.txt"):
        image_candidates = list(src_images.glob(label_path.stem + ".*"))
        if not image_candidates:
            continue

        output_lines = []

        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue

            old_id = int(parts[0])

            if old_id not in CLASS_MAP:
                skipped += 1
                continue

            new_id = CLASS_MAP[old_id]
            coords = list(map(float, parts[1:]))

            # Standard YOLO bbox
            if len(coords) == 4:
                output_lines.append(
                    f"{new_id} " + " ".join(f"{v:.6f}" for v in coords)
                )

            # Polygon -> bounding box
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                xs = coords[0::2]
                ys = coords[1::2]

                x_min = max(0.0, min(xs))
                x_max = min(1.0, max(xs))
                y_min = max(0.0, min(ys))
                y_max = min(1.0, max(ys))

                w = x_max - x_min
                h = y_max - y_min

                if w > 0 and h > 0:
                    xc = (x_min + x_max) / 2
                    yc = (y_min + y_max) / 2

                    output_lines.append(
                        f"{new_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                    )
                    converted += 1

        if not output_lines:
            continue

        image_path = image_candidates[0]
        new_stem = f"arduino_{label_path.stem}"

        shutil.copy2(
            image_path,
            dst_images / f"{new_stem}{image_path.suffix}"
        )

        (dst_labels / f"{new_stem}.txt").write_text(
            "\n".join(output_lines) + "\n"
        )

        copied += 1

    print(f"{src_split} -> {dst_split}")
    print(f"  Images copied:       {copied}")
    print(f"  Polygon conversions: {converted}")
    print(f"  Unsupported labels:  {skipped}")

print("\nArduino dataset merge complete.")
