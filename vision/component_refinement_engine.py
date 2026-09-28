import os
import cv2
import torch
from torch import nn
from torchvision import transforms, models

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(_ROOT, "models", "component_classifier", "best.pt")
CLASSES_PATH = os.path.join(_ROOT, "models", "component_classifier", "classes.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

_model = None
_classes = None


def load_classifier():
    global _model, _classes

    if _model is not None:
        return _model, _classes

    import json

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        _classes = json.load(f)

    _model = models.mobilenet_v3_small(weights=None)
    _model.classifier[3] = nn.Linear(
        _model.classifier[3].in_features,
        len(_classes)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        _model.load_state_dict(checkpoint["model_state_dict"])
    else:
        _model.load_state_dict(checkpoint)

    _model = _model.to(DEVICE)
    _model.eval()

    return _model, _classes


def refine_crop(crop):
    if crop is None or crop.size == 0:
        return None

    model, classes = load_classifier()

    image = _transform(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, class_id = torch.max(probabilities, dim=1)

    return {
        "class_id": int(class_id.item()),
        "class_name": classes[int(class_id.item())],
        "confidence": float(confidence.item())
    }


def refine_detection(image, box, padding=12, scale=3.0):
    height, width = image.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    new_width = max(1, int(crop.shape[1] * scale))
    new_height = max(1, int(crop.shape[0] * scale))

    enlarged = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    result = refine_crop(enlarged)

    if result is None:
        return None

    result["box"] = [x1, y1, x2, y2]
    result["crop"] = enlarged

    return result