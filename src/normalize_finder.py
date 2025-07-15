from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

dataset = datasets.ImageFolder(root="../data/bean_224", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

mean = 0.0
std = 0.0
nb_samples = 0.0

for data, _ in tqdm(loader):
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)  # [B, C, H*W]
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f"Mean: {mean}")
print(f"Std: {std}")
