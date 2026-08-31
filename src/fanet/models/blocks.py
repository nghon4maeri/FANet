import torch
import torch.nn as nn

""" Squeeze and Excitation block """
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

""" 3x3->3x3 Residual block """
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4

""" Mixpool block: Merging the image features and the mask """
class MixPool(nn.Module):
    """MixPool with configurable gating.

    gate:
        "binary" - hard threshold (original FANet, zero gradient to fmask)
        "ste"    - hard forward + straight-through gradient to fmask
        "soft"   - soft gating max(fmask, m_fg), full gradient
    dual_path:
        False - mask m is [B,1,H,W] foreground only
        True  - mask m is [B,2,H,W] = [m_fg, m_bg]; confident background
                suppresses activation (kept = keep * (1 - m_bg))
    """
    def __init__(self, in_c, out_c, gate="binary", dual_path=False):
        super(MixPool, self).__init__()

        assert gate in ("binary", "ste", "soft"), f"unknown gate {gate}"
        self.gate = gate
        self.dual_path = dual_path

        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        fmask = self.fmask(x)  # soft attention, differentiable

        m = nn.MaxPool2d((m.shape[2]//x.shape[2], m.shape[3]//x.shape[3]))(m)
        m_fg = m[:, 0:1]
        m_bg = m[:, 1:2] if (self.dual_path and m.shape[1] > 1) else torch.zeros_like(m_fg)

        if self.gate == "binary":
            fmask_g = (fmask > 0.5).float()
        elif self.gate == "ste":
            fmask_g = (fmask > 0.5).float() + fmask - fmask.detach()
        else:  # soft
            fmask_g = fmask

        keep = torch.maximum(fmask_g, m_fg)              # fg path (OR semantics)
        if self.dual_path:
            keep = keep * (1.0 - m_bg)                   # bg suppression

        x1 = x * keep
        x1 = self.conv1(x1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], axis=1)
        return x
