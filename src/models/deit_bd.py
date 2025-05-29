import timm

def deit_bd(num_classes=2):
    return timm.create_model("deit_base_distilled_patch16_224", pretrained=True, num_classes=2)
