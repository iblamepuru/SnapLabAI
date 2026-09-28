from pathlib import Path
import shutil

root = Path("datasets/snaplab_master")
backup = Path("datasets/snaplab_master/_arduino_label_backup")

fixed = 0
changed_files = 0

for label_path in (root / "labels" / "train").glob("arduino_*.txt"):

    lines = label_path.read_text(errors="ignore").splitlines()
    new_lines = []
    file_changed = False

    for line in lines:

        if not line.strip():
            continue

        p = line.split()

        if len(p) != 5:
            new_lines.append(line)
            continue

        cls = p[0]
        xc, yc, w, h = map(float, p[1:])

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        # Clip box boundaries to valid YOLO range.
        nx1 = max(0.0, min(1.0, x1))
        ny1 = max(0.0, min(1.0, y1))
        nx2 = max(0.0, min(1.0, x2))
        ny2 = max(0.0, min(1.0, y2))

        nw = nx2 - nx1
        nh = ny2 - ny1

        if nw <= 0 or nh <= 0:
            print("WARNING: unusable box:", label_path.name, line)
            continue

        nxc = (nx1 + nx2) / 2
        nyc = (ny1 + ny2) / 2

        new_line = (
            f"{cls} "
            f"{nxc:.6f} {nyc:.6f} "
            f"{nw:.6f} {nh:.6f}"
        )

        if new_line != line:
            fixed += 1
            file_changed = True

        new_lines.append(new_line)

    if file_changed:

        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            label_path,
            backup / label_path.name
        )

        label_path.write_text(
            "\n".join(new_lines) + "\n",
            encoding="utf-8"
        )

        changed_files += 1

print("\n========== ARDUINO LABEL REPAIR ==========")
print("Boxes repaired :", fixed)
print("Files changed  :", changed_files)
print("Backup folder  :", backup)
