# importing libraries
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class FaceDataset(Dataset):

    def __init__(self, root_dir):
        # root_dir = "spatial_faces" or "fft_faces"
        self.root_dir = root_dir
        self.samples = []

        # build list 
        # label: 0 = original, 1 = manipulated
        for label_name in ["original", "manipulated"]:
            label_dir = os.path.join(root_dir, label_name)
            label = 0 if label_name == "original" else 1

            for video_folder in os.listdir(label_dir):
                video_path = os.path.join(label_dir, video_folder)

                for img_name in os.listdir(video_path):
                    img_path = os.path.join(video_path, img_name)
                    self.samples.append((img_path, label))

        # transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # number of samples
        self.n_samples = len(self.samples)

    # support indexing: dataset[i]
    def __getitem__(self, index):
        img_path, label = self.samples[index]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label

    # len(dataset)
    def __len__(self):
        return self.n_samples
