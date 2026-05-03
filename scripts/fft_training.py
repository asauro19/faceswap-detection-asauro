import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from fft_dataset import FFTFaceDataset
from tqdm import tqdm 

# load FFT faces directly from split folders
train_dataset = FFTFaceDataset("/home/adrianna/Downloads/faceswap-research/faceswap-detection-asauro/fft_faces/train")
val_dataset   = FFTFaceDataset("/home/adrianna/Downloads/faceswap-research/faceswap-detection-asauro/fft_faces/val")
test_dataset  = FFTFaceDataset("/home/adrianna/Downloads/faceswap-research/faceswap-detection-asauro/fft_faces/test")

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# dataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# load ResNet18
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# training
epochs = 5

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

torch.save({
    "epoch": epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "accuracy": accuracy,
}, "fft_resnet18.pth")

print("FFT Model saved")
