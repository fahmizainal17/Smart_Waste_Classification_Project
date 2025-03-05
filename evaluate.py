# evaluate.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from data_loader import val_loader, class_names

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load("trashnet_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        # images should have shape [batch_size, 3, 224, 224]
        if images.ndim == 3:
            images = images.unsqueeze(0)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total
print(f"Validation Accuracy: {val_acc:.4f}")
