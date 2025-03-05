# app.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import gradio as gr
from PIL import Image
from torchvision import transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 6)
model.load_state_dict(torch.load("trashnet_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = transform_pipeline(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = nn.functional.softmax(outputs, dim=1)[0]
    return {class_names[i]: float(probs[i]) for i in range(6)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="TrashNet Classifier",
    description="Upload an image of trash (cardboard, glass, metal, paper, plastic, or general trash)."
)

if __name__ == "__main__":
    demo.launch()
