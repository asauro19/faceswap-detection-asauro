# importing libraries
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class FFTFaceDataset(Dataset):  

    def __init__(self, root_dir):
        # root_dir = "fft_faces"   
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

        # transform (same as RGB)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.n_samples = len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        # load FFT images as true grayscale 
        img = Image.open(img_path).convert("L")

        # convert grayscale → RGB (so model still gets 3 channels)
        img = img.convert("RGB")

        img = self.transform(img)

        return img, label

    def __len__(self):
        return self.n_samples
