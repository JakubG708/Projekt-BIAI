import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageColor


class ColorDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images = images_dir
        self.labels = labels_dir
        self.transform = transform
        self.data = []

        # Zbieramy wszystkie pary (obraz, kolor)
        for filename in os.listdir(self.labels):
            if filename.endswith('.txt'):
                path = os.path.join(self.labels, filename)
                with open(path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) != 2:
                            continue
                        img_name, hex_color = parts
                        rgb_color = ImageColor.getrgb(hex_color)
                        normalized_rgb = torch.tensor([c / 255.0 for c in rgb_color], dtype=torch.float32)
                        
                        # Dodajemy każdą parę (obraz, kolor) jako osobny rekord
                        self.data.append((img_name, normalized_rgb))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, target_color = self.data[idx]
        img_path = os.path.join(self.images, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target_color




