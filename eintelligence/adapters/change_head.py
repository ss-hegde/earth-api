import torch
import torch.nn as nn
import torch.nn.functional as F

class ChangeUNetHead(nn.Module):
    """
    UNet-like decoder for concatenated temporal inputs (e.g. [t1_bands, t2_bands])
    """

    def __init__(self, feat_channels = (64, 128, 256, 512), out_channels=1):
        super().__init__()
        C1, C2, C3, C4 = feat_channels
        def upconv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            )
        self.up3 = upconv(C4 + C3, C3)
        self.up2 = upconv(C3 + C2, C2)
        self.up1 = upconv(C2 + C1, C1)
        self.up0 = nn.Sequential(nn.Conv2d(C1, C1//2, kernel_size=3, padding=1), nn.ReLU(True), nn.Conv2d(C1//2, out_channels, kernel_size=1))

    def forward(self, feats):
        """
        feats: list of feature tensors from backbone, ordered low to high resolution
        """
        c1, c2, c3, c4 = feats  # assume 4 levels
        x = F.interpolate(c4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up3(nn.functional.relu(torch.cat([x, c3], dim=1)))
        x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up2(nn.functional.relu(torch.cat([x, c2], dim=1)))
        x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up1(nn.functional.relu(torch.cat([x, c1], dim=1)))
        H, W = c1.shape[-2]*4, c1.shape[-1]*4
        x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
        x = self.up0(x)
        return x  # (B, out_channels, H, W)