import os
import random
import logging
from datasets import load_dataset
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    """
    Muat turun dataset 'garythung/trashnet' & ambil subset 300 imej saja,
    kemudian bahagikan 80% -> train dan 20% -> val.
    Simpan imej dalam folder data/train/<kelas> dan data/val/<kelas>.
    """
    logger.info("Mula muat turun dataset garythung/trashnet ...")
    full_dataset = load_dataset("garythung/trashnet", split="train")  # ±2527 imej

    logger.info(f"Jumlah imej asal: {len(full_dataset)}")
    # 1) Hadkan jumlah imej
    max_images = 300
    logger.info(f"Mengehadkan dataset kepada {max_images} imej sahaja.")
    if len(full_dataset) > max_images:
        # Shuffle dahulu supaya random subset
        full_dataset = full_dataset.shuffle(seed=42)
        # Ambil subset terawal (300 imej)
        full_dataset = full_dataset.select(range(max_images))

    # Sekarang dataset hanya ada max 300 imej
    logger.info(f"Sekarang dataset mempunyai {len(full_dataset)} imej.")

    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    # Root folders
    root_dir = "data"
    train_dir = os.path.join(root_dir, "train")
    val_dir   = os.path.join(root_dir, "val")

    logger.info("Mencipta folder data/train/<kelas> dan data/val/<kelas> ...")
    for c in class_names:
        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(val_dir, c), exist_ok=True)

    # Bahagi 80/20
    random.seed(42)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    train_size = int(0.8 * len(full_dataset))
    train_indices = set(indices[:train_size])
    val_indices   = set(indices[train_size:])

    logger.info(f"Train size: {train_size}, Val size: {len(full_dataset)-train_size}")

    # Loop & simpan imej
    for i, example in enumerate(full_dataset):
        img_data = example["image"]
        label_id = example["label"]
        label_name = class_names[label_id]

        # Convert to PIL
        if not isinstance(img_data, Image.Image):
            img_data = Image.fromarray(img_data)

        # Tentukan folder
        if i in train_indices:
            folder_path = os.path.join(train_dir, label_name)
        else:
            folder_path = os.path.join(val_dir, label_name)

        save_name = f"{label_name}_{i}.jpg"
        save_path = os.path.join(folder_path, save_name)
        img_data.save(save_path)

    logger.info(">>> Subset dataset (300 imej) siap di folder data/train & data/val ✅")

if __name__ == "__main__":
    main()
