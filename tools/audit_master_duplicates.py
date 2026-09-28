from pathlib import Path
from hashlib import md5
from collections import defaultdict

root = Path("datasets/snaplab_master/images")

hashes = defaultdict(list)

for split in ["train", "val", "test"]:
    folder = root / split

    for img in folder.glob("*"):
        if img.is_file():
            h = md5(img.read_bytes()).hexdigest()
            hashes[h].append((split, img.name))

duplicates = {
    h: files
    for h, files in hashes.items()
    if len(files) > 1
}

print("========== MASTER DUPLICATE AUDIT ==========")
print("Total images :", sum(len(list((root / s).glob("*"))) for s in ["train","val","test"]))
print("Unique hashes:", len(hashes))
print("Duplicate groups:", len(duplicates))
print()

cross_split = []

for h, files in duplicates.items():
    splits = set(x[0] for x in files)

    if len(splits) > 1:
        cross_split.append((h, files))

print("Cross-split duplicate groups:", len(cross_split))

if cross_split:
    print()
    print("First 20 cross-split duplicates:")
    for h, files in cross_split[:20]:
        print(files)
else:
    print("No cross-split duplicates found.")

print()
print("========== AUDIT COMPLETE ==========")
