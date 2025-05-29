import timm

def deit_b(num_classes=2):
    return timm.create_model("deit_base_patch16_224", pretrained=True, num_classes=2)
