import torch
from torchvision.models import resnet50, ResNet50_Weights

print("Loading ResNet50...")

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

print("Exporting TorchScript model...")

example_input = torch.randn(1, 3, 224, 224)

traced_model = torch.jit.trace(model, example_input)
traced_model.save("models/resnet50.pt")

print("Model saved successfully:")
print("models/resnet50.pt")