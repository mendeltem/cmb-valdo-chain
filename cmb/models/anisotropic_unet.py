"""Anisotropic 3D U-Net -- the tournament winner (``a03-aniso``).

Why this architecture
---------------------
The training grid is 0.5 x 0.5 x 1.0 mm, but two of the three VALDO cohorts were acquired
with 3-4 mm slices. Through-plane, most voxels are therefore interpolated, not measured.
A standard 3x3x3 kernel mixes sharp in-plane information with those interpolated values
from the very first layer.

This network trusts the in-plane direction first:

    level 0   convolutions (1, 3, 3), pooling (1, 2, 2)    -> grid becomes 1 x 1 x 1 mm
    level 1-3 ordinary 3x3x3 convolutions, pooling 2       -> full 3D context
    bottleneck

3D context is still needed: a microbleed ends after a few millimetres, a vessel continues
into the neighbouring slices.

Measured (VALDO, 5-fold CV, 57 cases, pooled lesion F1, 40 epochs with early stopping):
0.404 versus 0.245 for the 3D attention U-Net, paired bootstrap +0.159 [+0.091, +0.234].
It wins in all three cohorts, not only in the thick-slice ones.

``attention_gates=True`` adds the gates of the attention U-Net to all four skips
(``a11-anisoatt``); with 40 epochs this showed no measurable gain.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AttentionGate, ConvBlock


class AnisotropicUNet3D(nn.Module):
    """Input ``(B, in_channels, D, H, W)`` -> logits ``(B, 1, D, H, W)``.

    ``D`` must be divisible by 8, ``H`` and ``W`` by 16 (wrap in ``PadToMultiple``).
    """

    IN_PLANE_KERNEL = (1, 3, 3)
    IN_PLANE_POOL = (1, 2, 2)

    def __init__(self, in_channels: int, base_channels: int = 32,
                 attention_gates: bool = False) -> None:
        super().__init__()
        c = base_channels

        # Encoder
        self.enc0 = ConvBlock(in_channels, c, kernel=self.IN_PLANE_KERNEL)
        self.enc1 = ConvBlock(c, 2 * c)
        self.enc2 = ConvBlock(2 * c, 4 * c)
        self.enc3 = ConvBlock(4 * c, 8 * c)
        self.bottleneck = ConvBlock(8 * c, 16 * c)
        self.pool_in_plane = nn.MaxPool3d(self.IN_PLANE_POOL)
        self.pool = nn.MaxPool3d(2)

        # Decoder (transposed convolutions mirror the poolings)
        self.up3 = nn.ConvTranspose3d(16 * c, 8 * c, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(8 * c, 4 * c, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(4 * c, 2 * c, 2, stride=2)
        self.up0 = nn.ConvTranspose3d(2 * c, c, self.IN_PLANE_POOL, stride=self.IN_PLANE_POOL)
        self.dec3 = ConvBlock(16 * c, 8 * c)
        self.dec2 = ConvBlock(8 * c, 4 * c)
        self.dec1 = ConvBlock(4 * c, 2 * c)
        self.dec0 = ConvBlock(2 * c, c, kernel=self.IN_PLANE_KERNEL)

        # One gate per skip, deepest first (same order as the decoder loop below).
        self.gates = (nn.ModuleList([AttentionGate(n, n) for n in (8 * c, 4 * c, 2 * c, c)])
                      if attention_gates else None)
        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.enc0(x)
        skip1 = self.enc1(self.pool_in_plane(skip0))
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        h = self.bottleneck(self.pool(skip3))

        decoder_levels = ((self.up3, self.dec3, skip3), (self.up2, self.dec2, skip2),
                          (self.up1, self.dec1, skip1), (self.up0, self.dec0, skip0))
        for level, (up, dec, skip) in enumerate(decoder_levels):
            upsampled = up(h)
            if self.gates is not None:
                skip = self.gates[level](upsampled, skip)
            h = dec(torch.cat([upsampled, skip], dim=1))
        return self.head(h)
