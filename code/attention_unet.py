#!/usr/bin/env python3
"""we3: AttentionUNet3D with searchable architecture (in_channels, depth, gates optional).

Rebuilt from an earlier project state (versuche/attention-unet/attention_unet.py), original there
UNTOUCHED. Rules:
  - in_channels parameter (1 or 2), not hardwired
  - depth as a channel list: [b,2b,4b] = 3 levels, [b,2b,4b,8b] = 4, [b,2b,4b,8b,16b] = 5
  - attention gates switchable off (gates=False -> plain U-Net)
  - fix 9 Sep stays: gate g and skip x have the SAME number of channels at the same stage.
  - depth 4, basis 32, gates on, 1 channel MUST yield the same parameter count as the
    original (2 channels) -- the only difference is the input conv.

Normalization (Claude 11 Sep): norm="batch" (default, as before -- old models load
unchanged) or "instance" (InstanceNorm3d affine; independent of the batch size 4
and of the fact that training batches are 60% lesion patches, but the inference
tiles are not). In the gate psi with "instance" there is no norm, just conv with bias -> sigmoid.

Axis convention (z, y, x) = NIfTI (D, H, W). Padding to multiples of 2^tiefe
in forward, cropped back afterward (as in the original).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm3d(art, c):
    if art == "batch":
        return nn.BatchNorm3d(c)
    if art == "instance":
        return nn.InstanceNorm3d(c, affine=True)
    raise ValueError(f"norm {art!r} (batch/instance)")


class AttentionGate3D(nn.Module):
    def __init__(self, f_g, f_x, f_int=None, norm="batch"):
        super().__init__()
        f_int = f_int or min(f_g, f_x) // 2
        self.w_g = nn.Sequential(nn.Conv3d(f_g, f_int, 1, bias=False), norm3d(norm, f_int))
        self.w_x = nn.Sequential(nn.Conv3d(f_x, f_int, 1, bias=False), norm3d(norm, f_int))
        if norm == "batch":
            self.psi = nn.Sequential(nn.Conv3d(f_int, 1, 1, bias=False), nn.BatchNorm3d(1), nn.Sigmoid())
        else:
            self.psi = nn.Sequential(nn.Conv3d(f_int, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, g, x):
        h = F.relu(self.w_g(g) + self.w_x(x), inplace=True)
        h = F.interpolate(h, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return x * self.psi(h)


class Block(nn.Module):
    """(3x3x3-Conv -> Norm -> ReLU) * 2"""
    def __init__(self, ein, aus, dropout=0.0, norm="batch"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ein, aus, 3, padding=1, bias=False), norm3d(norm, aus), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(aus, aus, 3, padding=1, bias=False), norm3d(norm, aus), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def kanalliste(basis, tiefe):
    """[b, 2b, 4b, ...] with 'tiefe' entries (tiefe 3/4/5)."""
    return [basis * 2 ** i for i in range(tiefe)]


class AttentionUNet3D(nn.Module):
    """Input (B, in_channels, D, H, W) -> logits (B, 1, D, H, W).

    D, H, W are padded to the next multiple of 2**tiefe.
    """
    def __init__(self, in_channels=1, basis=32, tiefe=4, gates=True, dropout=0.0, norm="batch"):
        super().__init__()
        assert tiefe in (3, 4, 5), f"tiefe {tiefe} not allowed (3/4/5)"
        ch = kanalliste(basis, tiefe)          # encoder channels per stage
        m = ch[-1] * 2                          # middle (original: Block(b*8, b*16))
        self.e = nn.ModuleList([Block(in_channels, ch[0], dropout, norm)] +
                               [Block(ch[i], ch[i + 1], dropout, norm) for i in range(tiefe - 1)])
        self.mitte = Block(ch[-1], m, dropout, norm)
        self.u = nn.ModuleList([nn.ConvTranspose3d(m, ch[-1], 2, stride=2)] +
                               [nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2)
                                for i in range(tiefe - 2, -1, -1)])
        self.d = nn.ModuleList([Block(m, ch[-1], dropout, norm)] +
                               [Block(ch[i + 1], ch[i], dropout, norm) for i in range(tiefe - 2, -1, -1)])
        if gates:
            # Fix 9 Sep: g (upsampled) and x (skip) have the same number of channels.
            self.ag = nn.ModuleList([AttentionGate3D(c, c, norm=norm) for c in reversed(ch)])
        else:
            self.ag = None
        self.aus = nn.Conv3d(ch[0], 1, 1)
        self.pool = nn.MaxPool3d(2)
        self.tiefe = tiefe
        self.in_channels = in_channels
        self.div = 2 ** tiefe

    def forward(self, x):
        D, H, W = x.shape[-3:]
        pd = ((self.div - D % self.div) % self.div,
              (self.div - H % self.div) % self.div,
              (self.div - W % self.div) % self.div)
        if any(pd):
            x = F.pad(x, (0, pd[2], 0, pd[1], 0, pd[0]))
        skips = [self.e[0](x)]
        h = skips[0]
        for i in range(1, self.tiefe):
            h = self.e[i](self.pool(h))
            skips.append(h)
        h = self.mitte(self.pool(h))
        for i in range(self.tiefe):
            g = self.u[i](h)
            s = skips[self.tiefe - 1 - i]
            if self.ag is not None:
                s = self.ag[i](g, s)
            h = self.d[i](torch.cat([g, s], 1))
        y = self.aus(h)
        if any(pd):
            y = y[..., :D, :H, :W]
        return y
