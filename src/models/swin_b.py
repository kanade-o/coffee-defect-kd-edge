import timm
import torch

def swin_b(num_classes=2, path="/home/sota/research/sotaohnuma/coffee/src/models/weights/swin_base_patch4_window7_224.pth"):
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=num_classes)
    state_dict = torch.load(path, map_location="cpu")
    # 分類ヘッドは学習し直す想定なので除外
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    model.load_state_dict(filtered_state_dict, strict=False)
    return model
