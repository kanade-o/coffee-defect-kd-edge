import os
import timm
import torch
SAVE_DIR = "/home/sota/research/sotaohnuma/coffee/src/models/"

def timm_model(model_name: str, num_classes=2):
    save_path = os.path.join(SAVE_DIR, f"{model_name}.pth")
    
    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict, strict=False)
    return model

a = timm_model("resnet18")
print(a)
