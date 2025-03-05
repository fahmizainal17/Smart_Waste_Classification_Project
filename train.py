"""
Workflow:
-----------
This script trains a fine-tuned ResNet18 model on a custom image classification dataset.
It performs the following steps:

1. Setup and Initialization:
   - Imports necessary libraries including PyTorch, torchvision, logging, and tqdm for progress visualization.
   - Configures logging to output information messages.
   - Determines the computation device: uses Apple’s MPS if available, otherwise defaults to CPU.

2. Model Preparation:
   - Loads a pre-trained ResNet18 model using the latest weights.
   - Modifies the final fully connected layer to output predictions corresponding to the number of classes in the dataset.
   - Moves the model to the selected device.

3. Training Loop:
   - Iterates over the training DataLoader which is assumed to return properly batched images ([B, 3, 224, 224]) and labels ([B]).
   - Performs a forward pass through the model.
   - Computes the loss using CrossEntropyLoss.
   - Backpropagates the loss and updates the model parameters using the Adam optimizer.
   - Tracks running loss and accuracy.
   - Saves the model every 20 batches (overwriting the previous saved model).

4. Epoch Logging and Final Save:
   - Logs the average loss and accuracy for each epoch.
   - Saves the final trained model to "trashnet_resnet18.pth".
   
Usage:
   - Ensure that the data_loader module correctly provides train_loader, val_loader, and class_names.
   - If the DataLoader does not stack the samples correctly, consider using a custom collate function as shown above.
   - Adjust hyperparameters such as epochs, learning rate, and save frequency as needed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from data_loader import train_loader, val_loader, class_names
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable MPS (for Apple GPU) or fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load pre-trained ResNet18 using the new weights parameter
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 5
save_every = 20  # Save model every 20 batches

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")):
        # DataLoader should return images in shape [B, 3, 224, 224] and labels in shape [B]
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

        if (batch_idx + 1) % save_every == 0:
            torch.save(model.state_dict(), "trashnet_resnet18.pth")
            logger.info(f"Model saved after batch {batch_idx+1} ✅")

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

torch.save(model.state_dict(), "trashnet_resnet18.pth")
logger.info("Final model saved as trashnet_resnet18.pth ✅")
