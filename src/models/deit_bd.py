import timm
import torch

def deit_bd(num_classes=2, path="/home/sota/research/sotaohnuma/coffee/src/models/weights/deit_base_distilled_patch16_224.pth"):
    model = timm.create_model("deit_base_distilled_patch16_224", pretrained=False, num_classes=num_classes)
    state_dict = torch.load(path, map_location="cpu")
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("head.") or k.startswith("head_dist."))
    }
    model.load_state_dict(filtered_state_dict, strict=False)

    return model
