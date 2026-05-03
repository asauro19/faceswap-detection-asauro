import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from rgb_dataset import FaceDataset
from splits import make_faceforensics_splits
from tqdm import tqdm 


# load faces
dataset = FaceDataset("/home/adrianna/Downloads/faceswap-research/faceswap-detection-asauro/spatial_faces")  

# official FaceForensics++ train/val/test split
split_dir = "/home/adrianna/Downloads/faceswap-research/FaceForensics/dataset/splits"
train_dataset, val_dataset, test_dataset = make_faceforensics_splits(dataset, split_dir)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# dataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# load ResNet18
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)  # binary classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# training 
epochs = 5

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}  (Remaining: {epochs - (epoch+1)})")
    model.train()
    running_loss = 0.0

    # tqdm progress bar for batches
    for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", ncols=100):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")


# evaluation

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"\nTest Accuracy: {accuracy:.4f}")

#save lol
torch.save({
    "epoch": epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "accuracy": accuracy, 
}, "rgb_resnet18.pth")
print("Model saved")
