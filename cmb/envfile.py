"""``cmb.env()``: the paths of the tutorials, read from ``.env`` in the repository root (no extra dependency).

Lines ``NAME=value`` (``#`` comments, optional quotes). Relative paths resolve against the repository root, not the current
directory, so the tutorials run from anywhere. A variable set in the shell environment wins over ``.env``.
Names are those of cmb_3d_unet_experiments: CMB_INPUT_DIR, CMB_LABEL_DIR, CMB_MODEL_DIR, CMB_OUTPUT_DIR, CMB_SCAN (+ CMB_*).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_KEYS_SUFFIX = ("_DIR", "_SCAN", "_FILE")


def env(path: str = None) -> SimpleNamespace:
    path = path or os.path.join(REPO, ".env")
    values = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    for k in list(values):
        values[k] = os.environ.get(k, values[k])
        if (k.endswith(PATH_KEYS_SUFFIX) or k == "CMB_SCAN") and values[k] and not os.path.isabs(values[k]):
            values[k] = os.path.join(REPO, values[k])
    return SimpleNamespace(**values)
