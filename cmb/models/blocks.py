"""Small building blocks shared by all networks.

Conventions used throughout the package
---------------------------------------
* Tensors are ``(batch, channels, depth, height, width)`` = ``(B, C, z, y, x)``.
* Every network maps ``(B, in_channels, D, H, W)`` to logits ``(B, 1, D, H, W)``.
  Apply ``torch.sigmoid`` to obtain probabilities.
* Normalisation is InstanceNorm everywhere. Reason: training batches hold only four
  patches, 60 % of which contain a lesion, while inference tiles almost never do.
  BatchNorm statistics learned on the former do not fit the latter (measured
  2026-09-11: InstanceNorm F1 0.245 vs BatchNorm 0.215 for the attention U-Net).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def instance_norm(channels: int) -> nn.InstanceNorm3d:
    """InstanceNorm with learnable scale and shift."""
    return nn.InstanceNorm3d(channels, affine=True)


class ConvBlock(nn.Sequential):
    """Two ``Conv3d -> InstanceNorm -> LeakyReLU`` layers.

    ``kernel`` may be anisotropic, e.g. ``(1, 3, 3)`` convolves only within a slice.
    ``dilation`` widens the receptive field without losing resolution.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel: Sequence[int] = (3, 3, 3), dilation: int = 1) -> None:
        padding = tuple(dilation * (k // 2) for k in kernel)

        def layer(c_in: int) -> list[nn.Module]:
            return [nn.Conv3d(c_in, out_channels, tuple(kernel), padding=padding,
                              dilation=dilation, bias=False),
                    instance_norm(out_channels),
                    nn.LeakyReLU(0.01, inplace=True)]

        super().__init__(*layer(in_channels), *layer(out_channels))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018) for one skip connection.

    ``gate`` is the upsampled decoder feature, ``skip`` the encoder feature of the same
    level. The gate produces one weight in [0, 1] per voxel that scales ``skip``.
    """

    def __init__(self, gate_channels: int, skip_channels: int,
                 inter_channels: Optional[int] = None) -> None:
        super().__init__()
        inter_channels = inter_channels or min(gate_channels, skip_channels) // 2
        self.project_gate = nn.Sequential(nn.Conv3d(gate_channels, inter_channels, 1, bias=False),
                                          instance_norm(inter_channels))
        self.project_skip = nn.Sequential(nn.Conv3d(skip_channels, inter_channels, 1, bias=False),
                                          instance_norm(inter_channels))
        # No normalisation on the single-channel attention map: InstanceNorm over one
        # channel would remove exactly the information the map is supposed to carry.
        self.attention = nn.Sequential(nn.Conv3d(inter_channels, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        mixed = F.relu(self.project_gate(gate) + self.project_skip(skip), inplace=True)
        mixed = F.interpolate(mixed, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return skip * self.attention(mixed)


class PadToMultiple(nn.Module):
    """Pad ``(D, H, W)`` so every axis is a multiple of ``multiple``; crop the output back.

    U-Nets halve the grid several times, so each axis must be divisible by
    ``2 ** number_of_poolings``. Our patch is ``(38, 62, 94)`` voxels -- none of these is.
    ``minimum`` enforces a lower bound (Swin-UNETR needs at least 64 per axis).
    ``select`` extracts the tensor from networks that return a list (U-Net++).
    """

    def __init__(self, net: nn.Module, multiple: Sequence[int],
                 minimum: Sequence[int] = (0, 0, 0),
                 select: Optional[Callable[[object], torch.Tensor]] = None) -> None:
        super().__init__()
        self.net = net
        self.multiple = tuple(multiple)
        self.minimum = tuple(minimum)
        self.select = select

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-3:]
        target = [max(-(-size[i] // self.multiple[i]) * self.multiple[i], self.minimum[i])
                  for i in range(3)]
        pad = [target[i] - size[i] for i in range(3)]
        if any(pad):
            # F.pad takes the LAST axis first: (x_left, x_right, y_left, y_right, z_left, z_right)
            x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
        out = self.net(x)
        if self.select is not None:
            out = self.select(out)
        return out[..., :size[0], :size[1], :size[2]]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
