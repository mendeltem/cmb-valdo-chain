"""Pretrained weights of the paper (``cmb get-model``), downloaded on first use and checked by SHA-256.

    cmb get-model valdo_t2           # -> ~/.cache/cmb-valdo-chain/valdo_t2/ (or $CMB_MODEL_CACHE)
    cmb get-model --list

Only weights trained on PUBLIC data are listed here, and the training inputs of every file were checked (not its name):

* ``valdo_t2`` -- the stage-1 U-Net of the paper trained on ALL 72 VALDO 2021 Task 2 scans (``dev/manifest_valdo_all72.json``,
  ids sub-101 ... sub-327, 236 CMB; ``cmb.training.full_data train``, seeds 42 and 1, 60 epochs, state after the last
  epoch). These two files are byte-identical to the ones the paper used as the Phase 2 starting point ("the VALDO-72
  U-Net", variants FT / FTn and the n = 0 arm). No in-house scan, threshold or calibration entered them: the fixed
  post-processing cell (threshold 0.3, >= 2 mm3, <= 2100 voxels) was chosen on the VALDO inner folds.

* ``valdo_stage2`` -- a stage-2 candidate classifier trained on VALDO alone (added after the paper, release v1.1.0):
  ``cmb.stage2.final_classifier`` on the 278 out-of-fold candidates of the 57 VALDO cross-validation cases. It is not
  the paper's frozen-V classifier (that one was retrained in memory for every scoring run and never saved, see
  PAPER_VERSION.md); ``cmb predict --stage2`` uses it.

Files are release assets of github.com/mendeltem/cmb-valdo-chain (tag below), not part of the git tree: the weights carry
the licence of the data they were derived from.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
from typing import Dict

TAG = "v1.1.0"  # release of main with both stages; the paper tag paper-v3-2026-11 holds the stage-1 files too
BASE = f"https://github.com/mendeltem/cmb-valdo-chain/releases/download/{TAG}/"
VALDO_CITE = (
    "Sudre CH et al. Where is VALDO? VAscular Lesions Detection and segmentatiOn challenge at MICCAI 2021. "
    "Medical Image Analysis (2024) 103029, doi:10.1016/j.media.2023.103029; data: "
    "doi:10.5281/zenodo.4520773"
)

REGISTRY: Dict[str, Dict] = {
    "valdo_t2": dict(
        kind="stage1",
        files=[  # (asset name, sha256, file inside the model folder)
            (
                "valdo_t2_seed42_modell.pt",
                "b9301ccb867c4177fe942067cb463d237411e6abc7349e282e1d2520493cfda1",
                "seed42/modell.pt",
            ),
            (
                "valdo_t2_seed42_meta.json",
                "f6e4ee69c672a9f945e2de6ce8a5476de3074592a0e551e4352290f57dec1122",
                "seed42/meta.json",
            ),
            (
                "valdo_t2_seed1_modell.pt",
                "f0cc5495b3fa6441ef62b859c0d7fc31f07a686c686b30de967d3a8bc9fc755f",
                "seed1/modell.pt",
            ),
            (
                "valdo_t2_seed1_meta.json",
                "84f792cba7b9582adc87165de95e1c44cb4982afa0f201c1b22079815f5f571d",
                "seed1/meta.json",
            ),
        ],
        sequence="T2*-GRE",
        trained_on="VALDO 2021 Task 2, all 72 public training scans (SABRE, RSS, ALFA; 236 CMB); nothing else",
        recipe="a03-aniso U-Net (image + FRST, width 32), 60 epochs, 800 patches/epoch, state after the last epoch; "
        "two members (seeds 42 and 1), averaged",
        cell=dict(threshold=0.3, min_mm3=2.0, max_voxels=2100),
        licence="CC BY-NC-SA 4.0 (derived from the VALDO data): non-commercial use, share alike, attribution",
        cite=VALDO_CITE,
    ),
    "valdo_stage2": dict(
        kind="stage2",
        files=[
            (
                "valdo_stage2_classifier.pt",
                "ef179dca588ab522a27344c94f6001a7c01a4c701b27655e3808216302d6d4ef",
                "classifier.pt",
            ),
            (
                "valdo_stage2_classifier.json",
                "ab3663ca0c608f87051e35194e465dd495ec8f2304d64857842ff2ec94f700c8",
                "final_classifier.json",
            ),
        ],
        sequence="T2*-GRE",
        trained_on="the 278 stage-1 candidates (106 on a CMB) of the 57 VALDO cross-validation cases, out of fold; nothing else",
        recipe="cmb.stage2.final_classifier: small 3D CNN (16-32-64 channels) on 16 x 32 x 32 voxel cubes of image, FRST and "
        "stage-1 probability, 3 members, 40 epochs; threshold 0.55 by 5-fold CV by case (pooled out-of-fold "
        "lesion F1 0.614 on VALDO); candidates: probability > 0.15, >= 1 mm3, CSF kept",
        cell=dict(threshold=0.55),
        licence="CC BY-NC-SA 4.0 (derived from the VALDO data): non-commercial use, share alike, attribution",
        cite=VALDO_CITE,
    ),
}


def cache_dir() -> str:
    # the per-user download cache of the README (like pip's), not a data location
    return os.environ.get("CMB_MODEL_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "cmb-valdo-chain"))  # noqa: TID251


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, target: str, log) -> None:
    tmp = target + ".part"
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, 1 << 20)
    os.replace(tmp, target)


def get_model(name: str = "valdo_t2", target: str | None = None, log=None, base: str | None = None) -> str:
    """Download (once) and verify a registered model; returns the model folder (``<folder>/<member>/modell.pt`` +
    ``meta.json`` per member, the layout ``cmb.training.full_data`` writes). ``CMB_MODEL_URL`` overrides the download
    base (a mirror or a local ``file://`` folder)."""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if name not in REGISTRY:
        raise SystemExit(f"unknown model {name!r}; available: {', '.join(REGISTRY)}")
    entry = REGISTRY[name]
    folder = target or os.path.join(cache_dir(), name)
    base = base or os.environ.get("CMB_MODEL_URL", BASE)
    if not base.endswith("/"):
        base += "/"
    for asset, digest, rel in entry["files"]:
        local = os.path.join(folder, rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if os.path.exists(local) and (digest is None or sha256(local) == digest):
            continue
        _download(base + asset, local, log)
        if digest is not None and sha256(local) != digest:
            bad = sha256(local)
            os.remove(local)
            raise SystemExit(f"{asset}: SHA-256 mismatch ({bad} != {digest}); file removed")
    info = {k: v for k, v in entry.items() if k != "files"}
    info.update(name=name, tag=TAG, members=sorted({os.path.dirname(r) for _, _, r in entry["files"] if "/" in r}))
    json.dump(info, open(os.path.join(folder, "model.json"), "w"), indent=1)
    return folder


def describe(name: str) -> str:
    e = REGISTRY[name]
    return (
        f"{name} ({e['kind']}, {e['sequence']}, release {TAG})\n  trained on: {e['trained_on']}\n  recipe: {e['recipe']}\n"
        f"  licence: {e['licence']}\n  cite: {e['cite']}"
    )
