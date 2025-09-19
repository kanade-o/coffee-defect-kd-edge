import torch
import timm
import torch.nn as nn

def tiny_vit_np(
    num_classes: int = 2,
    path: str = "/home/sota/research/sotaohnuma/coffee/src/models/weights/tiny_vit_np.pth",
):
    model = timm.create_model("tiny_vit_5m_224", pretrained=False, num_classes=num_classes)

    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model", "net")):
        sd = sd.get("state_dict", sd.get("model", sd))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # 分類ヘッド系は除外（distilled系のhead_distも念のため除外）
    filtered = {
        k: v
        for k, v in sd.items()
        if not (k.startswith("head") or k.startswith("head_dist") or k.startswith("classifier") or k.startswith("fc"))
    }
    model.load_state_dict(filtered, strict=False)
    return model
