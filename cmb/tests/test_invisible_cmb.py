# -*- coding: utf-8 -*-
"""CPU unit tests for cmb.analysis.invisible_cmb (wc10, 2026-11-05).

Synthetic volume: one dark blob with stage-1 probability 0 (must come out
INVISIBLE) and one with probability 0.9 (must come out VISIBLE). Covers the
per-lesion feature rows (size, slices, region, contrast, darkest-voxel offset,
edge distance, edge slices) and the visible/invisible split.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cmb.analysis.invisible_cmb import (  # noqa: E402
    C26, VISIBLE_CUTOFF, aggregate, brain_mask, edge_transform,
    lesion_features, region_majority, region_of,
)


def _synthetic():
    rng = np.random.default_rng(0)
    shape = (40, 40, 30)
    img = (0.5 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
    frst = (0.5 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
    prob = np.zeros(shape, np.float32)
    z, y, x = np.indices(shape)
    brain = ((x - 20) ** 2 / 30 ** 2 + (y - 20) ** 2 / 30 ** 2
             + (z - 15) ** 2 / 12 ** 2) < 1.0
    seg = np.where(brain, 3, 0).astype(np.int32)
    a = (x - 12) ** 2 + (y - 14) ** 2 + (z - 10) ** 2 < 9      # invisible blob
    b = (x - 28) ** 2 + (y - 26) ** 2 + (z - 20) ** 2 < 9      # visible blob
    lab = np.zeros(shape, int)
    lab[a] = 1
    lab[b] = 2
    img[a] = 0.2
    img[b] = 0.2
    frst[a] = 0.9
    frst[b] = 0.9
    prob[a] = 0.0
    prob[b] = 0.9
    spacing = np.array([0.5, 0.5, 1.0])
    return img, frst, prob, lab, seg, spacing


def test_two_lesions_found():
    img, frst, prob, lab, seg, sp = _synthetic()
    rows = lesion_features(img, frst, prob, lab, seg, sp)
    assert len(rows) == 2, f"expected 2 lesions, got {len(rows)}"


def test_visibility_split():
    img, frst, prob, lab, seg, sp = _synthetic()
    rows = lesion_features(img, frst, prob, lab, seg, sp)
    rA = next(r for r in rows if r["p_max"] == 0.0)
    rB = next(r for r in rows if r["p_max"] == 0.9)
    assert rA["visible"] is False, "p=0 lesion must be invisible"
    assert rB["visible"] is True, "p=0.9 lesion must be visible"
    assert VISIBLE_CUTOFF == 0.01


def test_geometry_features():
    img, frst, prob, lab, seg, sp = _synthetic()
    rows = lesion_features(img, frst, prob, lab, seg, sp)
    rA = next(r for r in rows if r["p_max"] == 0.0)
    # blob radius 2 -> 27 voxels, at least 1 slice, inside the ellipsoidal brain
    assert rA["n_voxels"] >= 1 and rA["n_slices"] >= 1
    assert rA["size_mm3"] == round(rA["n_voxels"] * 0.25, 2)
    assert rA["region"] == "lobar", rA
    assert 0 < rA["edge_mm"] < 50, rA
    assert rA["contrast"] is not None and rA["contrast"] > 1, rA
    # darkest voxel sits inside the 5-voxel window -> offset <= window size
    assert 0 <= rA["darkest_offset_mm"] <= 12, rA
    # lesion centre is far from the first/last 2 z-slices
    assert rA["edge_slices"] is False, rA


def test_edge_slices_flag():
    shape = (20, 20, 10)
    img = np.ones(shape, np.float32)
    frst = np.ones(shape, np.float32)
    prob = np.zeros(shape, np.float32)
    seg = np.ones(shape, np.int32) * 3
    lab = np.zeros(shape, int)
    lab[10:12, 10:12, 0:2] = 1                      # z=0,1: touches first 2 slices
    lab[10:12, 10:12, 8:10] = 2                     # z=8,9: touches last 2 slices
    rows = lesion_features(img, frst, prob, lab, seg, np.ones(3))
    assert len(rows) == 2
    assert rows[0]["edge_slices"] is True
    assert rows[1]["edge_slices"] is True


def test_region_of_and_brain_mask():
    assert region_of(3) == "lobar"
    assert region_of(4) == "csf_other"
    seg = np.zeros((8, 8, 8), int)
    seg[2:6, 2:6, 2:6] = 2
    assert brain_mask(seg).sum() == 64
    dt, brain = edge_transform(seg)                  # padded by 2 voxels
    assert brain.shape == tuple(s + 4 for s in seg.shape) and dt.shape == brain.shape
    assert dt[4 + 2, 4 + 2, 4 + 2] > 0               # deep inside: positive distance
    assert brain[2:2 + 8, 2:2 + 8, 2:2 + 8].shape == seg.shape


def test_aggregate_invisible_visible():
    rows = [
        dict(n_voxels=20, size_mm3=5.0, n_slices=2, region="lobar", contrast=1.5,
             darkest_offset_mm=0.3, edge_mm=10.0, edge_slices=False, p_max=0.0, visible=False),
        dict(n_voxels=30, size_mm3=7.5, n_slices=3, region="deep_grey", contrast=4.0,
             darkest_offset_mm=0.5, edge_mm=5.0, edge_slices=True, p_max=0.9, visible=True),
    ]
    s = aggregate(rows, "test")
    assert s["n_invisible"] == 1 and s["n_visible"] == 1, s
    assert set(s["invisible"]) >= {"contrast", "size_mm3", "edge_slices_share"}


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"{len(fns)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
