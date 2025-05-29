import torch.nn as nn
import torchvision.models as models

def resnet18(num_classes=2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
