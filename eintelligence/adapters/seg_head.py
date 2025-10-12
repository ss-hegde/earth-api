# eintelligence/adapters/seg_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNetHead(nn.Module):
    """
    Lightweight UNet-ish decoder that takes encoder feature pyramid (4 scales)
    and returns a 1-channel mask (vegetation vs not) at full resolution.
    """
    def __init__(self, feat_channels=(64,128,256,512), out_ch=1):
        super().__init__()
        C1,C2,C3,C4 = feat_channels

        def up(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.up3 = up(C4 + C3, 256)
        self.up2 = up(256 + C2, 128)
        self.up1 = up(128 + C1, 64)
        self.up0 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1)
        )

    def forward(self, feats):
        C1,C2,C3,C4 = feats  # shapes: 1/4,1/8,1/16,1/32 resolutions
        x = F.interpolate(C4, size=C3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, C3], dim=1)
        x = self.up3(x)

        x = F.interpolate(x, size=C2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, C2], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, size=C1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, C1], dim=1)
        x = self.up1(x)

        # back to full resolution: assume input was 4x larger than C1
        H = C1.shape[-2]*4
        W = C1.shape[-1]*4
        x = F.interpolate(x, size=(H,W), mode="bilinear", align_corners=False)
        x = self.up0(x)
        return x  # logits [B,1,H,W]
