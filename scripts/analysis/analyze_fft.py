import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from fft_dataset import FFTFaceDataset



# 1. Load checkpoint 

ckpt = torch.load(
    "scripts/fft_resnet18.pth",
    map_location="cpu",
    weights_only=False   # REQUIRED for older checkpoints
)

print("Epoch trained:", ckpt["epoch"])
print("Test accuracy saved:", ckpt["test_accuracy"])
print("Confusion matrix saved:\n", ckpt["confusion_matrix"])



# 2. Rebuild the model

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# 3. Load test dataset

test_dataset = FFTFaceDataset(
    "/home/adrianna/Downloads/faceswap-research/faceswap-detection-asauro/scripts/frequency_faces/test"
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# 4. Run inference

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



# 5. Compute metrics

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = (all_preds == all_labels).mean()
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)

print("\n=== Recomputed Test Accuracy ===")
print(acc)

print("\n=== Confusion Matrix ===")
print(cm)

print("\n=== Classification Report ===")
print(report)
