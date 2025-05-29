import timm

def timm_model(name: str, num_classes=2):
    return timm.create_model(name, pretrained=True, num_classes=2)
