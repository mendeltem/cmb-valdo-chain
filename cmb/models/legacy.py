"""Load checkpoints written by the old scripts (``../code/netze.py``) into the new models.

The old code used German attribute names, so the state-dict keys differ although the
tensors are identical. ``convert_state_dict`` renames the keys; nothing else changes.
"""
from __future__ import annotations

import re
from typing import Dict

import torch

# (pattern, replacement), applied in order to every key.
_RULES = [
    (r"^netz\.", "net."),                              # PadToMultiple wrapper
    (r"\.mitte\.net\.", ".bottleneck."),
    (r"\.e(\d)\.net\.", r".enc\1."),
    (r"\.d(\d)\.net\.", r".dec\1."),
    (r"\.u(\d)\.", r".up\1."),
    (r"\.aus\.", ".head."),
    (r"\.ag\.(\d)\.w_g\.", r".gates.\1.project_gate."),
    (r"\.ag\.(\d)\.w_x\.", r".gates.\1.project_skip."),
    (r"\.ag\.(\d)\.psi\.", r".gates.\1.attention."),
]


def convert_state_dict(legacy: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted = {}
    for key, tensor in legacy.items():
        new_key = key
        for pattern, replacement in _RULES:
            new_key = re.sub(pattern, replacement, new_key)
        converted[new_key] = tensor
    return converted


def load_legacy_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load ``modell.pt`` from a tournament fold folder. Raises if any key is left over."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(convert_state_dict(state), strict=True)
    return model
