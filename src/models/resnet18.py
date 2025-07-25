import torch
import torch.nn as nn
import torchvision.models as models

def resnet18(num_classes=2, path="/home/sota/research/sotaohnuma/coffee/src/models/weights/resnet18.pth"):
    model = models.resnet18(pretrained=False)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
