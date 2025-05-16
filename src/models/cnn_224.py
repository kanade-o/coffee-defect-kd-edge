import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN224(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super(CNN224, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # [B, 3, 224, 224] → [B, 16, 224, 224]
        self.pool = nn.MaxPool2d(2, 2)                          # → [B, 16, 112, 112]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # → [B, 32, 112, 112] → pool → [B, 32, 56, 56]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # → [B, 64, 56, 56] → pool → [B, 64, 28, 28]
        
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → [B, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # → [B, 32, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # → [B, 64, 28, 28]
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

