import torch
import torch.nn as nn
import torchvision.models as models

def efficientnet_b0(num_classes=2, path="/home/sota/research/sotaohnuma/coffee/src/models/weights/efficientnet_b0.pth"):
    model = models.efficientnet_b0(pretrained=False)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
