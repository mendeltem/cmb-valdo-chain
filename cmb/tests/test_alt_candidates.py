"""cmb.stage2.alt_candidates on a synthetic two-case grid (CPU, no patient data).

    python -m cmb.tests.test_alt_candidates          (run from the repository root)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def build(tmp: str):
    rng = np.random.RandomState(0)
    grid, base = os.path.join(tmp, "grid"), os.path.join(tmp, "base")
    affine = np.diag([0.5, 0.5, 1.0, 1.0])
    for k, case in enumerate(["sub-a", "sub-b"]):
        z, y, x = np.mgrid[0:24, 0:48, 0:48]
        brain = ((y - 24) ** 2 + (x - 24) ** 2 < 20 ** 2)
        image = np.where(brain, 0.4 + 0.02 * rng.randn(24, 48, 48), 0).astype(np.float32)
        unet_blob = ((z - 8) ** 2 + (y - 15) ** 2 + (x - 15) ** 2) <= 4          # the U-Net already finds this one
        missed = ((z - 16) ** 2 + (y - 32) ** 2 + (x - 30) ** 2) <= 4           # this one only the image shows
        image[unet_blob | missed] = 0.95
        from scipy import ndimage
        frst = ndimage.gaussian_filter(image, 1.0) * brain
        p = np.zeros_like(image); p[unet_blob] = 0.8; p[(z == 3) & (y == 24) & (x == 24)] = 0.1
        os.makedirs(f"{grid}/{case}", exist_ok=True); os.makedirs(f"{base}/fold{k}", exist_ok=True)
        for name, arr in (("image", image), ("frst", frst)):
            nib.save(nib.Nifti1Image(arr.transpose(2, 1, 0), affine), f"{grid}/{case}/{case}_{name}.nii.gz")
        nib.save(nib.Nifti1Image(np.round(p * 255).astype(np.uint8).transpose(2, 1, 0), affine), f"{base}/fold{k}/{case}_pred_proba.nii.gz")
    return grid, base


def run(args):
    from cmb.stage2 import alt_candidates
    sys.argv = ["alt"] + args
    alt_candidates.main()


def test_union_keeps_unet_values_and_adds_the_missed_blob():
    with tempfile.TemporaryDirectory() as tmp:
        grid, base = build(tmp)
        for mode in ("frst", "gmm", "both"):
            out = os.path.join(tmp, f"alt-{mode}")
            run(["--base", base, "--grid", grid, "--mode", mode, "--out", out, "--frst-quantile", "99.5"])
            for k, case in enumerate(["sub-a", "sub-b"]):
                old = np.asarray(nib.load(f"{base}/fold{k}/{case}_pred_proba.nii.gz").dataobj).transpose(2, 1, 0)
                new = np.asarray(nib.load(f"{out}/fold{k}/{case}_pred_proba.nii.gz").dataobj).transpose(2, 1, 0)
                assert (new >= old).all(), mode                                   # nothing the U-Net had is lowered
                assert new[8, 15, 15] == old[8, 15, 15] == 204, mode               # U-Net candidate unchanged (0.8 * 255)
                assert new[16, 32, 30] == 41 and old[16, 32, 30] == 0, (mode, new[16, 32, 30])   # missed blob enters at 0.16
                assert new[3, 24, 24] == old[3, 24, 24], mode                      # sub-threshold U-Net voxel untouched
                assert (new[old == 0] > 0).mean() < 0.02, mode                     # a few seeds, not the whole brain
            assert json.load(open(f"{out}/alt_candidates.json"))["cases"] == 2


def test_cap_keeps_the_strongest_components():
    from cmb.stage2.alt_candidates import cap_components
    new = np.zeros((20, 20, 10), bool); strength = np.zeros(new.shape, np.float32)
    for k, (x, v) in enumerate(((2, 0.2), (8, 0.9), (14, 0.5))):
        new[x:x + 2, 5:7, 4:6] = True; strength[x:x + 2, 5:7, 4:6] = v                 # three blobs of 8 voxels
    new[18, 18, 8] = True; strength[18, 18, 8] = 1.0                                     # a single strong voxel: too small
    kept = cap_components(new, strength, max_new=2, min_voxels=4)
    assert kept[8, 5, 4] and kept[14, 5, 4] and not kept[2, 5, 4] and not kept[18, 18, 8]
    assert cap_components(new, strength, 0, 4).sum() == 24


def test_capped_run_adds_at_most_k_components():
    with tempfile.TemporaryDirectory() as tmp:
        grid, base = build(tmp)
        out = os.path.join(tmp, "capped")
        run(["--base", base, "--grid", grid, "--mode", "gmm", "--out", out, "--max-new", "1"])
        from scipy import ndimage
        for k, case in enumerate(["sub-a", "sub-b"]):
            old = np.asarray(nib.load(f"{base}/fold{k}/{case}_pred_proba.nii.gz").dataobj)
            new = np.asarray(nib.load(f"{out}/fold{k}/{case}_pred_proba.nii.gz").dataobj)
            assert ndimage.label((new > 0) & (old == 0), structure=np.ones((3, 3, 3), bool))[1] <= 1
        assert json.load(open(f"{out}/alt_candidates.json"))["max_new"] == 1


def test_refuses_to_overwrite_the_base_run():
    with tempfile.TemporaryDirectory() as tmp:
        grid, base = build(tmp)
        try:
            run(["--base", base, "--grid", grid, "--mode", "frst", "--out", base])
        except SystemExit as e:
            assert "differ" in str(e)
        else:
            raise AssertionError("overwrote the base run")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__, flush=True)
    print(f"ALL {len(tests)} TESTS OK")
