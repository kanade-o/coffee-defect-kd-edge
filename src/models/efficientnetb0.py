import timm

def efficientnetb0(num_classes=2):
    return timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)

