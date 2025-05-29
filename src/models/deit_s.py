import timm

def deit_s(num_classes=2):
    return timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=2)
