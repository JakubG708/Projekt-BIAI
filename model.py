import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Warstwa 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Warstwa 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Warstwa 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 3),
            nn.Sigmoid()  # RGB w [0, 1]
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)