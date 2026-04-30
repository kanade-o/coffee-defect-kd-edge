import os
import timm
import torch

os.environ["TORCH_HOME"] = "/home/sota/research/sotaohnuma/.cache/torch"
SAVE_DIR = "/home/sota/research/sotaohnuma/coffee/src/"

def download_model(model_name: str) -> None:
    save_path = os.path.join(SAVE_DIR, f"{model_name}.pth")

    if not os.path.exists(save_path):
        try:
            print(f"Downloading pretrained weights for: {model_name}")
            model = timm.create_model(model_name, pretrained=False)
            torch.save(model.state_dict(), save_path)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Download failed for {model_name}: {e}")
            return

    try:
        model = timm.create_model(model_name, pretrained=False)
        state_dict = torch.load(save_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Loading failed for {model_name}: {e}")

if __name__ == "__main__":
    model_list = [
#        "resnet18", 
#        "resnet50", 
#        "efficientnet_b0", 
#        "deit_base_patch16_224", 
#        "deit_base_distilled_patch16_224",
#        "deit_small_patch16_224",
#        "swin_tiny_patch4_window7_224",
#        "swin_small_patch4_window7_224",
#        "swin_base_patch4_window7_224",
#        'tiny_vit_5m_224.dist_in22k',
#        'mobilenetv2_120d.ra_in1k'
        'mobilenetv2_050.lamb_in1k',
    ]
    
    for model in model_list:
        download_model(model)
    
    print("All models processed.")

