"""Network registry.

``build_model(name, in_channels)`` returns a module that accepts any patch size (padding
is handled by ``PadToMultiple``) and outputs logits of the same spatial size.

Only architectures that have been ported to this package are listed. The remaining
tournament entries still live in ``../code/netze.py`` and are ported next
(plain nnU-Net style, 2.5D, residual-SE, U-Net++, shallow dilated, Swin-UNETR, SegResNet).
"""
from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from .anisotropic_unet import AnisotropicUNet3D
from .blocks import PadToMultiple, count_parameters

# Depth is pooled three times (factor 8), height and width four times (factor 16).
_ANISO_MULTIPLE = (8, 16, 16)

MODELS: Dict[str, Callable[[int], nn.Module]] = {
    "aniso": lambda c: PadToMultiple(AnisotropicUNet3D(c), _ANISO_MULTIPLE),
    "aniso-attention": lambda c: PadToMultiple(AnisotropicUNet3D(c, attention_gates=True), _ANISO_MULTIPLE),
}

# Names used in the tournament result folders (``ergebnisse/turnier/<legacy name>``).
LEGACY_NAMES = {"a03-aniso": "aniso", "a11-anisoatt": "aniso-attention"}


def build_model(name: str, in_channels: int) -> nn.Module:
    name = LEGACY_NAMES.get(name, name)
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](in_channels)


__all__ = ["MODELS", "LEGACY_NAMES", "build_model", "count_parameters"]
