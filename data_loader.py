# data_loader.py
from datasets import load_dataset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, random_split
from PIL import Image

# 1. Extract – Load dataset from Hugging Face
dataset = load_dataset("garythung/trashnet")

# Define class labels for the 6 categories
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# 2. Transform – Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (ResNet input size)
    transforms.ToTensor(),          # Convert images to tensor (should produce [3, 224, 224])
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # Normalize images
])

def transform_example(example):
    image = example["image"]
    # If the image is already a PIL Image, convert it to RGB
    if not isinstance(image, list):
        image = image.convert("RGB")
    else:
        # If it's a list, check if its first element is a filepath or already an image.
        if len(image) > 0 and isinstance(image[0], str):
            image = Image.open(image[0]).convert("RGB")
        else:
            image = image[0].convert("RGB")
    example["image"] = transform(image)
    return example

# Apply the transformation function to the dataset
dataset = dataset.with_transform(transform_example)

# 3. Split the dataset into train and validation (80/20 split)
train_size = int(0.8 * len(dataset["train"]))
val_size = len(dataset["train"]) - train_size
train_data, val_data = random_split(dataset["train"], [train_size, val_size])

# 4. Define a collate function to stack images and labels into tensors
def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])  # Expected shape: [B, 3, 224, 224]
    labels = torch.tensor([item["label"] for item in batch])
    return images, labels

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

print("Dataset loaded and transformed successfully ✅")
