import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageColor
from skimage import color
import numpy as np

class ColorDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.data = []

        # Test konwersji
        self._test_color_conversions()
        
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                path = os.path.join(labels_dir, filename)
                with open(path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) != 3:
                            continue
                        
                        img_name, hex_color, hex_color_secondary = parts
                        hex_color = hex_color.strip(',;')
                        hex_color_secondary = hex_color_secondary.strip(',;')
                        
                        try:
                            # Pierwszy kolor
                            rgb = ImageColor.getrgb(hex_color)
                            rgb_normalized = np.array(rgb, dtype=np.float32) / 255.0
                            lab = color.rgb2lab(rgb_normalized.reshape(1, 1, 3))[0, 0]
                            
                            # Drugi kolor
                            rgb_secondary = ImageColor.getrgb(hex_color_secondary)
                            rgb_secondary_normalized = np.array(rgb_secondary, dtype=np.float32) / 255.0
                            lab_secondary = color.rgb2lab(rgb_secondary_normalized.reshape(1, 1, 3))[0, 0]
                            
                            # Normalizacja LAB
                            lab_normalized = [
                                lab[0] / 100.0,
                                (lab[1] + 128) / 255.0,
                                (lab[2] + 128) / 255.0
                            ]
                            
                            lab_secondary_normalized = [
                                lab_secondary[0] / 100.0,
                                (lab_secondary[1] + 128) / 255.0,
                                (lab_secondary[2] + 128) / 255.0
                            ]
                            
                            self.data.append((
                                img_name, 
                                torch.tensor(lab_normalized, dtype=torch.float32),
                                torch.tensor(lab_secondary_normalized, dtype=torch.float32)
                            ))
                            
                        except ValueError as e:
                            print(f"Invalid color in line: {line.strip()} | Error: {str(e)}")
                            continue

    def _test_color_conversions(self):
        """Testy poprawności konwersji kolorów"""
        test_colors = [
            ("#FFFFFF", "White", [100.0, 0.0, 0.0]),
            ("#000000", "Black", [0.0, 0.0, 0.0]),
            ("#FF0000", "Red", [53.2408, 80.0925, 67.2032]),
            ("#00FF00", "Green", [87.7347, -86.1827, 83.1793])
        ]
        
        for hex_color, name, expected_lab in test_colors:
            rgb = ImageColor.getrgb(hex_color)
            rgb_normalized = np.array(rgb, dtype=np.float32) / 255.0
            lab = color.rgb2lab(rgb_normalized.reshape(1, 1, 3))[0, 0]
            print(f"\nTEST {name}:")
            print(f"HEX: {hex_color}")
            print(f"RGB (0-1): {rgb_normalized}")
            print(f"LAB (calculated): {lab}")
            print(f"LAB (expected): {expected_lab}")



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, target_color, target_color_secondary = self.data[idx]
        img_path = os.path.join(self.images_dir, img_name)  # Zmiana z self.images na self.images_dir
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        combined_targets = torch.cat([target_color, target_color_secondary])
        return image, combined_targets







