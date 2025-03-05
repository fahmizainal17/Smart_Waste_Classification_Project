# app.py
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import gradio as gr
from PIL import Image
import os

# ===== Setup Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Memulakan Gradio app untuk Klasifikasi Sampah...")

    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Pastikan model.pth wujud
    if not os.path.exists("model.pth"):
        logger.error("model.pth tidak dijumpai! Sila jalankan train.py dahulu.")
        raise FileNotFoundError("model.pth tidak dijumpai")

    # Load model
    logger.info("Muat ResNet18 & bebankan parameter model.pth ...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Transforms
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

        logger.info("Inference dijalankan - keputusan dihasilkan.")
        return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    # Gradio interface
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=6),
        title="Trash Classification (Demo)",
        description="Upload gambar sampah: cardboard, glass, metal, paper, plastic, atau trash. (Latihan ringkas!)"
    )

    logger.info("Aplikasi Gradio kini dilancar! Sila semak URL local.")
    demo.launch(share=True)

if __name__ == "__main__":
    main()
