import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from rgb_dataset import RGBFaceDataset  



# 1. Load checkpoint

ckpt = torch.load("rgb_resnet18.pth", map_location="cpu")

print("Epoch trained:", ckpt["epoch"])
print("Test accuracy saved:", ckpt["test_accuracy"])
print("Confusion matrix saved:\n", ckpt["confusion_matrix"])



# 2. Rebuild the model (classification)

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# 3. Load test dataset

test_dataset = RGBFaceDataset(
    "/home/adrianna/Downloads/faceswap-research/faceswap-detection-asauro/scripts/spatial_faces/test"
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# 4. Run inference (classification)

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



# ---------------------------------------------------------
# 6. FEATURE VECTOR EXTRACTION (added section)
# ---------------------------------------------------------

print("\n=== Extracting 512‑dim RGB feature vectors ===")

# Rebuild model with Identity head for embeddings
feature_model = models.resnet18(weights="IMAGENET1K_V1")
feature_model.fc = nn.Identity()   # <-- outputs 512‑dim features

feature_model.load_state_dict(ckpt["model_state_dict"], strict=False)
feature_model.eval()
feature_model = feature_model.to(device)

all_features = []
all_labels_feat = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)

        feats = feature_model(imgs)      # shape: [batch, 512]
        feats = feats.cpu().numpy()

        all_features.append(feats)
        all_labels_feat.append(labels.numpy())

all_features = np.vstack(all_features)
all_labels_feat = np.concatenate(all_labels_feat)

print("RGB Feature matrix shape:", all_features.shape)
print("Labels shape:", all_labels_feat.shape)

np.save("rgb_features.npy", all_features)
np.save("rgb_feature_labels.npy", all_labels_feat)

print("Saved rgb_features.npy and rgb_feature_labels.npy")
