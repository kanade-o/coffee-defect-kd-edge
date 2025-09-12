import timm
from ptflops import get_model_complexity_info

# timmから全モデルリストを取得
all_timm_models = timm.list_models(pretrained=False)

# 事前に指定したモデル
base_models = [
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "deit_base_patch16_224",
    "deit_base_distilled_patch16_224",
    "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
]

# MobileNet系とTinyViT系をフィルタで抽出
mobilenet_models = [m for m in all_timm_models if m.startswith("mobilenet")]
tinyvit_models = [m for m in all_timm_models if m.startswith("tiny_vit")]

# すべて統合
model_list = base_models + mobilenet_models + tinyvit_models

# 入力サイズ
input_size = (3, 224, 224)

def to_gmacs(macs_str: str) -> float:
    """ptflopsの文字列(MMAC, GMAC, etc)をfloat[GMAC]に変換"""
    macs_str = macs_str.strip().upper()
    if macs_str.endswith("GMAC"):
        return float(macs_str.replace("GMAC", "").strip())
    if macs_str.endswith("MMAC"):
        return float(macs_str.replace("MMAC", "").strip()) / 1000
    if macs_str.endswith("KMAC"):
        return float(macs_str.replace("KMAC", "").strip()) / 1e6
    if macs_str.endswith("MAC"):
        return float(macs_str.replace("MAC", "").strip()) / 1e9
    raise ValueError(f"Unknown unit in {macs_str}")

def to_million(params_str: str) -> float:
    """パラメータ数 (M単位に統一)"""
    params_str = params_str.strip().upper()
    if params_str.endswith("M"):
        return float(params_str.replace("M", "").strip())
    if params_str.endswith("K"):
        return float(params_str.replace("K", "").strip()) / 1000
    if params_str.endswith("G"):
        return float(params_str.replace("G", "").strip()) * 1000
    if params_str.isdigit():
        return float(params_str) / 1e6
    raise ValueError(f"Unknown unit in {params_str}")

# 各モデルを順に計算
for model_name in model_list:
    try:
        model = timm.create_model(model_name, pretrained=False)
        macs_str, params_str = get_model_complexity_info(
            model, input_size, as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        macs_g = to_gmacs(macs_str)
        params_m = to_million(params_str)

        print(f"モデル: {model_name}")
        print(f"  計算量 : {macs_g:.2f} GMAC")
        print(f"  パラメータ数 : {params_m:.2f} M")
        print("-" * 40)
    except Exception as e:
        print(f"モデル {model_name} の計算に失敗: {e}")
        print("-" * 40)
