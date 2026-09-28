from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main():
    root = Path("datasets/component_classifier")
    output_dir = Path("models/component_classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        root / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root / "val",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print("=" * 65)
    print("SNAPLAB AI — COMPONENT CLASSIFIER TRAINING")
    print("=" * 65)
    print("Device:", device)
    print("Classes:", len(train_dataset.classes))
    print("Train crops:", len(train_dataset))
    print("Validation crops:", len(val_dataset))
    print("Batch size:", 32)
    print("Input size:", "224x224")
    print()

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(train_dataset.classes)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    best_accuracy = 0.0

    for epoch in range(15):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch + 1:02d}/15 "
            f"Loss={train_loss:.4f} "
            f"TrainAcc={train_accuracy:.4f} "
            f"ValAcc={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "input_size": 224,
                    "best_val_accuracy": best_accuracy
                },
                output_dir / "best.pt"
            )

            print(
                f"Best model saved: {best_accuracy:.4f}"
            )

    with open(
        output_dir / "classes.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            train_dataset.classes,
            f,
            indent=2
        )

    print()
    print("=" * 65)
    print("CLASSIFIER TRAINING COMPLETE")
    print("=" * 65)
    print("Classes:", len(train_dataset.classes))
    print(
        "Best validation accuracy:",
        f"{best_accuracy:.4f}"
    )
    print(
        "Model:",
        output_dir / "best.pt"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()