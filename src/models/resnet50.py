import torch.nn as nn
from torchvision.models import resnet50

def resnet(weight_path, num_classes=2):
    model = resnet50(pretrained=True)
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
