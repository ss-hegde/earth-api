import torch
import torch.nn as nn
import timm 

class ResNetBackbone(nn.Module):
    """
    Small, fast encoder using timm's resnet18, modified to accept C channels (e.g., 4: B02,B03,B04,B08).
    Outputs feature maps at 1/32 resolution (like standard ResNet encoders).
    """

    def __init__(self, in_channels: int = 4, name: str = "resnet18"):
        super().__init__()
        self.encoder = timm.create_model(name, pretrained=True, in_chans=in_channels, features_only=True, out_indices=(1,2,3,4))

    def forward(self, x):
        feats = self.encoder(x)  # list: [C1,H/4,W/4], [C2,H/8,W/8], [C3,H/16,W/16], [C4,H/32,W/32]
        return feats