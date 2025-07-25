import os
import timm
import torch
import torch.nn as nn

SAVE_DIR = "/home/sota/research/sotaohnuma/coffee/src/models/"

def replace_classifier(model, num_classes):
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError("This model is NOT cnns")

def timm_model(model_name: str, num_classes=2):
    save_path = os.path.join(SAVE_DIR, f"{model_name}.pth")

    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(save_path, map_location="cpu")

    ignored_prefixes = ["fc.", "head.", "classifier."]
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in ignored_prefixes)
    }
    model.load_state_dict(filtered_state_dict, strict=False)

    replace_classifier(model, num_classes)

    return model

if __name__ == "__main__":
    model = timm_model("resnet18")
    print(model)
    model = timm_model("resnet50")
    print(model)
    model = timm_model("efficientnet_b0")
    print(model)

