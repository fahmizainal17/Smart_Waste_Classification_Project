# train.py
import os
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset

##############################
#         LOGGING
##############################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

##############################
#        SETTINGS
##############################
DATA_DIR     = "data"  # Folder "data" dari download_trashnet.py
TRAIN_SUBSET = 200     # Ambil 200 imej untuk training (supaya cepat)
VAL_SUBSET   = 50      # Ambil 50 imej untuk validation
EPOCHS       = 2       # Latih 2 epoch

def main():
    logger.info("Mula proses training ResNet18 ...")

    # 1) Setup data folder
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    # 2) Transforms
    logger.info("Sediakan transforms & load dataset penuh ...")
    common_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_train_dataset = datasets.ImageFolder(train_dir, transform=common_transforms)
    full_val_dataset   = datasets.ImageFolder(val_dir,   transform=common_transforms)

    logger.info(f"Train folder total: {len(full_train_dataset)} imej.")
    logger.info(f"Val folder total:   {len(full_val_dataset)} imej.")

    # 3) Subset agar cepat
    random.seed(42)
    train_indices = list(range(len(full_train_dataset)))
    random.shuffle(train_indices)
    train_indices = train_indices[:TRAIN_SUBSET]

    val_indices = list(range(len(full_val_dataset)))
    random.shuffle(val_indices)
    val_indices = val_indices[:VAL_SUBSET]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset   = Subset(full_val_dataset,   val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

    class_names = full_train_dataset.classes
    logger.info(f"Kelas: {class_names}")

    # 4) Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Guna device: {device}")

    # 5) Model
    logger.info("Muat ResNet18 pra-latih & ubah suai output layer ...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model.to(device)

    # 6) Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 7) Training loop
    logger.info(f"Bermula training untuk {EPOCHS} epoch ...")
    for epoch in range(EPOCHS):
        logger.info(f"---- EPOCH {epoch+1} ----")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 2 == 0:
                logger.debug(f"Batch {batch_idx+1} - Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total if total else 0.0
        epoch_acc  = correct / total if total else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total if val_total else 1
        val_acc = val_correct / val_total if val_total else 0.0

        logger.info(f"[Epoch {epoch+1}/{EPOCHS}] "
                    f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 8) Simpan model
    torch.save(model.state_dict(), "model.pth")
    logger.info(">>> Model disimpan sebagai 'model.pth' ✅")

if __name__ == "__main__":
    main()
