import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from fft_dataset import FFTFaceDataset
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# Load FFT datasets

train_dataset = FFTFaceDataset(
    "/home/adrianna/Downloads/faceswap-research/faceswap-detection-asauro/scripts/spatial_faces/train"
)
val_dataset = FFTFaceDataset(
    "/home/adrianna/Downloads/faceswap-research/faceswap-detection-asauro/scripts/spatial_faces/val"
)
test_dataset = FFTFaceDataset(
    "/home/adrianna/Downloads/faceswap-research/faceswap-detection-asauro/scripts/spatial_faces/test"
)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# DataLoaders

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

# Model setup

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 5

# Training loop

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", ncols=100):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Training Loss: {avg_loss:.4f}")


    # Validation
    
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.4f}")


# Final Test Evaluation

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_accuracy = (all_preds == all_labels).mean()
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix + Precision/Recall/F1
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report (Precision, Recall, F1):")
print(classification_report(all_labels, all_preds, digits=4))

# Save model
torch.save({
    "epoch": epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "test_accuracy": test_accuracy,
    "confusion_matrix": cm,
}, "fft_resnet18.pth")

print("FFT Model saved successfully.")
