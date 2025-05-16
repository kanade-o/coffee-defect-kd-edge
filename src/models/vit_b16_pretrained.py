import torch, torch.nn as nn
from torchvision.models import vit_b_16

def vit_b16_pretrained(weight_path: str, num_classes: int):
    model = vit_b_16()                       # ランダム初期化
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)             # 重みを注入
    # 最後の分類ヘッドをデータセットに合わせて差し替え
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

