from pathlib import Path

root = Path("datasets/snaplab_master")

fixed = 0
changed_files = 0

EPS = 1e-6

for split in ["train", "val"]:

    label_dir = root / "labels" / split

    for label_path in label_dir.glob("arduino_*.txt"):

        lines = label_path.read_text(errors="ignore").splitlines()
        output = []
        file_changed = False

        for line in lines:

            if not line.strip():
                continue

            p = line.split()

            if len(p) != 5:
                output.append(line)
                continue

            cls = p[0]

            xc, yc, w, h = map(float, p[1:])

            # Convert YOLO center representation to corners
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            # Clip to image boundaries with a tiny safety margin
            x1 = max(0.0, min(1.0 - EPS, x1))
            y1 = max(0.0, min(1.0 - EPS, y1))
            x2 = max(EPS, min(1.0 - EPS, x2))
            y2 = max(EPS, min(1.0 - EPS, y2))

            # Safety check
            if x2 <= x1:
                x2 = min(1.0 - EPS, x1 + EPS)

            if y2 <= y1:
                y2 = min(1.0 - EPS, y1 + EPS)

            # Convert back to YOLO format
            new_xc = (x1 + x2) / 2
            new_yc = (y1 + y2) / 2
            new_w = x2 - x1
            new_h = y2 - y1

            # Final numerical clamp
            new_xc = max(EPS, min(1.0 - EPS, new_xc))
            new_yc = max(EPS, min(1.0 - EPS, new_yc))
            new_w = max(EPS, min(1.0, new_w))
            new_h = max(EPS, min(1.0, new_h))

            new_line = (
                f"{cls} "
                f"{new_xc:.8f} {new_yc:.8f} "
                f"{new_w:.8f} {new_h:.8f}"
            )

            # Validate the newly generated box
            values = list(map(float, new_line.split()[1:]))

            if (
                values[0] - values[2] / 2 < -EPS or
                values[0] + values[2] / 2 > 1.0 + EPS or
                values[1] - values[3] / 2 < -EPS or
                values[1] + values[3] / 2 > 1.0 + EPS
            ):
                print("WARNING: still invalid:", label_path)
                output.append(line)
                continue

            if new_line != line:
                fixed += 1
                file_changed = True

            output.append(new_line)

        if file_changed:
            label_path.write_text(
                "\n".join(output) + "\n",
                encoding="utf-8"
            )
            changed_files += 1

print()
print("========== ROBUST ARDUINO LABEL SANITIZATION ==========")
print("Boxes modified :", fixed)
print("Files modified :", changed_files)
print("Splits         : train + val")
print("Status         : COMPLETE")
