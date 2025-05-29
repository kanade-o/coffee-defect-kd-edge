import torch, torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def vit_b16(num_classes=2):
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

