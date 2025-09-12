import torch
import timm
import torch.nn as nn

def mobilenetv2(
    num_classes: int = 2,
    path: str = "/home/sota/research/sotaohnuma/coffee/src/models/weights/mobilenetv2_120d.ra_in1k.pth",
):
    # ヘッドを所望クラス数で作成
    model = timm.create_model("mobilenetv2_120d", pretrained=False, num_classes=num_classes)

    # 重みを読み込み（state_dict/モデル直書きの両方に耐性）
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model", "net")):
        sd = sd.get("state_dict", sd.get("model", sd))
    # DDPなどの 'module.' 接頭辞を除去
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # 分類ヘッドは除外して読み込み
    filtered = {k: v for k, v in sd.items() if not (k.startswith("classifier") or k.startswith("head"))}
    model.load_state_dict(filtered, strict=False)
    return model
