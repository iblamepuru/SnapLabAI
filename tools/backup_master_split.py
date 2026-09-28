from pathlib import Path
import shutil
import datetime

root = Path("datasets/snaplab_master")
backup = Path("datasets/snaplab_master_split_backup")

backup.mkdir(exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

for file in [
    root / "data.yaml",
    root / "labels" / "train",
    root / "labels" / "val",
]:

    if file.is_file():
        shutil.copy2(file, backup / f"{timestamp}_{file.name}")

print("========== SPLIT BACKUP ==========")
print("Backup directory:", backup)
print("Timestamp:", timestamp)
print("Status: READY")
