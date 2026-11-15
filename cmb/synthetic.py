"""Tiny synthetic T2*-like scans for the self-test and the test suite (no real data).

Each phantom is a skull-stripped "brain" (ellipsoid, zero background) with a smooth bias field, a few dark spheres
(microbleeds, labelled) and dark tubes (vessels, not labelled), on an anisotropic grid like a clinical T2* scan.

    python -m cmb.synthetic OUT --n 6            # OUT/raw/<id>_T2starw.nii.gz, <id>_cmb.nii.gz, <id>_brainmask.nii.gz
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def phantom(seed: int, shape=(64, 64, 24), spacing=(0.9, 0.9, 2.5), n_cmb: int = 3):
    rng = np.random.RandomState(seed)
    sp = np.array(spacing)
    g = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing, strict=True)], indexing="ij"), -1)
    centre = (np.array(shape) - 1) * sp / 2
    radii = centre * np.array([0.85, 0.85, 0.8])
    brain = (((g - centre) / radii) ** 2).sum(-1) <= 1
    img = 600 + 40 * rng.randn(*shape)
    bias = 1 + 0.25 * (g[..., 0] - centre[0]) / centre[0]
    lab = np.zeros(shape, np.uint8)
    inside = np.argwhere(brain & (((g - centre) / (radii * 0.7)) ** 2).sum(-1).__le__(1))
    for _ in range(n_cmb):
        c = inside[rng.randint(len(inside))] * sp
        r = rng.uniform(1.2, 2.5)
        ball = ((g - c) ** 2).sum(-1) <= r**2
        img[ball] = 150 + 20 * rng.randn(int(ball.sum()))
        lab[ball] = 1
    for _ in range(2):  # vessels: dark tubes along y
        c = inside[rng.randint(len(inside))] * sp
        tube = ((g[..., 0] - c[0]) ** 2 + (g[..., 2] - c[2]) ** 2) <= 0.8**2
        img[tube & brain & (lab == 0)] = 250
    img = np.where(brain, img * bias, 0).astype(np.float32)
    lab = (lab & brain).astype(np.uint8)
    affine = np.diag(list(spacing) + [1.0])
    affine[:3, 3] = -centre
    return img, lab, brain.astype(np.uint8), affine


def write_cohort(out: str, n: int = 6, seed0: int = 0, labels: bool = True):
    import nibabel as nib

    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    ids = []
    for k in range(n):
        case = f"syn-{k:03d}"
        img, lab, brain, aff = phantom(seed0 + k, n_cmb=0 if k == 0 else 1 + k % 3)
        nib.save(nib.Nifti1Image(img, aff), f"{raw}/{case}_T2starw.nii.gz")
        nib.save(nib.Nifti1Image(brain, aff), f"{raw}/{case}_brainmask.nii.gz")
        if labels:
            nib.save(nib.Nifti1Image(lab, aff), f"{raw}/{case}_cmb.nii.gz")
        ids.append(case)
    return raw, ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raw, ids = write_cohort(a.out, a.n, a.seed)
    print(f"{len(ids)} phantoms -> {raw}")
