import os
import json
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

ROOT = "datasets/component_classifier"
MODEL_PATH = "models/component_classifier/best.pt"
CLASSES_PATH = "models/component_classifier/classes.json"
IMAGE_SIZE = 224
BATCH_SIZE = 32


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    class_names = sorted(
        [
            name
            for name in os.listdir(os.path.join(ROOT, "test"))
            if os.path.isdir(os.path.join(ROOT, "test", name))
        ]
    )

    samples = []

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name in class_names:
        class_dir = os.path.join(ROOT, "test", class_name)

        if class_name not in classes:
            continue

        class_index = classes.index(class_name)

        for filename in os.listdir(class_dir):
            path = os.path.join(class_dir, filename)

            if os.path.isfile(path):
                ext = os.path.splitext(filename)[1].lower()

                if ext in extensions:
                    samples.append((path, class_index))

    if not samples:
        raise RuntimeError("No test images found.")

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            path, target = self.items[index]
            image = datasets.folder.default_loader(path)
            image = self.transform(image)
            return image, target

    dataset = TestDataset(samples, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            for target, prediction in zip(targets, predictions):
                target_id = target.item()
                class_total[target_id] += 1

                if prediction.item() == target_id:
                    class_correct[target_id] += 1

    accuracy = 100.0 * correct / total

    print()
    print("SNAPLAB AI — COMPONENT CLASSIFIER TEST")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Test crops: {total}")
    print(f"Total classes: {len(classes)}")
    print()

    print(f"OVERALL TEST ACCURACY: {accuracy:.2f}%")
    print()

    print("PER-CLASS ACCURACY")
    print("=" * 60)

    available = 0
    unavailable = 0

    for index, class_name in enumerate(classes):
        if class_total[index] > 0:
            available += 1
            class_accuracy = 100.0 * class_correct[index] / class_total[index]
            print(
                f"{class_name:35} "
                f"{class_accuracy:6.2f}% "
                f"({class_correct[index]}/{class_total[index]})"
            )
        else:
            unavailable += 1
            print(
                f"{class_name:35} "
                f"NO TEST DATA"
            )

    print()
    print("=" * 60)
    print(f"Classes with test data: {available}")
    print(f"Classes without test data: {unavailable}")
    print(f"Total test crops evaluated: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()