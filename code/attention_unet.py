#!/usr/bin/env python3
"""we3: AttentionUNet3D mit suchbarer Architektur (in_channels, Tiefe, Gates optional).

Umbau aus einem frueheren Projektstand (versuche/attention-unet/attention_unet.py), Original dort
UNANGEFASST. Regeln:
  - in_channels Parameter (1 oder 2), nicht hart verdrahtet
  - Tiefe als Kanalliste: [b,2b,4b] = 3 Ebenen, [b,2b,4b,8b] = 4, [b,2b,4b,8b,16b] = 5
  - Attention-Gates abschaltbar (gates=False -> reines U-Net)
  - Fix 09.09. bleibt: Gate g und Skip x haben auf derselben Stufe GLEICH viele Kanaele.
  - Tiefe 4, basis 32, gates an, 1 Kanal MUSS dieselbe Parameterzahl wie das
    Original (2 Kanaele) liefern -- der einzige Unterschied ist das Eingangskonv.

Normierung (Claude 11.09.): norm="batch" (Vorgabe, wie bisher -- alte Modelle laden
unveraendert) oder "instance" (InstanceNorm3d affine; unabhaengig von der Stapelgroesse 4
und davon, dass Trainingsstapel zu 60 % Laesions-Patches sind, die Inferenzkacheln aber
nicht). Im Gate psi bei "instance" keine Norm, nur Conv mit Bias -> Sigmoid.

Achsenkonvention (z, y, x) = NIfTI (D, H, W). Padding auf Vielfache von 2^tiefe
im forward, zurueckschneiden danach (wie im Original).
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
    """[b, 2b, 4b, ...] mit 'tiefe' Eintraegen (tiefe 3/4/5)."""
    return [basis * 2 ** i for i in range(tiefe)]


class AttentionUNet3D(nn.Module):
    """Eingang (B, in_channels, D, H, W) -> Logits (B, 1, D, H, W).

    D, H, W werden auf das naechste Vielfache von 2**tiefe gepadded.
    """
    def __init__(self, in_channels=1, basis=32, tiefe=4, gates=True, dropout=0.0, norm="batch"):
        super().__init__()
        assert tiefe in (3, 4, 5), f"tiefe {tiefe} nicht erlaubt (3/4/5)"
        ch = kanalliste(basis, tiefe)          # Encoder-Kanaele je Stufe
        m = ch[-1] * 2                          # Mitte (Original: Block(b*8, b*16))
        self.e = nn.ModuleList([Block(in_channels, ch[0], dropout, norm)] +
                               [Block(ch[i], ch[i + 1], dropout, norm) for i in range(tiefe - 1)])
        self.mitte = Block(ch[-1], m, dropout, norm)
        self.u = nn.ModuleList([nn.ConvTranspose3d(m, ch[-1], 2, stride=2)] +
                               [nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2)
                                for i in range(tiefe - 2, -1, -1)])
        self.d = nn.ModuleList([Block(m, ch[-1], dropout, norm)] +
                               [Block(ch[i + 1], ch[i], dropout, norm) for i in range(tiefe - 2, -1, -1)])
        if gates:
            # Fix 09.09.: g (hochgetastet) und x (Skip) haben gleich viele Kanaele.
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
