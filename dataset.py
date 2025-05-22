import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageColor
from skimage import color
import numpy as np

class ColorDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir  # Dodaj tę linię
        self.labels_dir = labels_dir  # Dodaj tę linię
        self.transform = transform    # Dodaj tę linię
        self.data = []
        
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                path = os.path.join(labels_dir, filename)
                with open(path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) != 2:
                            continue
                        img_name, hex_color = parts
                        
                        # Konwersja HEX do RGB
                        rgb = ImageColor.getrgb(hex_color)
                        rgb_array = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
                        
                        # Konwersja RGB do LAB
                        lab = color.rgb2lab(rgb_array)[0, 0]
                        
                        # Normalizacja LAB
                        lab_normalized = [
                            lab[0] / 100.0,
                            (lab[1] + 128) / 255.0,
                            (lab[2] + 128) / 255.0
                        ]
                        
                        self.data.append((img_name, torch.tensor(lab_normalized, dtype=torch.float32)))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, target_color = self.data[idx]
        img_path = os.path.join(self.images_dir, img_name)  # Zmiana z self.images na self.images_dir
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target_color







