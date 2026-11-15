"""wk5 (31.10.2026): cmb.analysis.mars_location -- the pre-specified MARS classes (Gregoire 2009) on a synthetic label map."""
import numpy as np

from cmb.analysis.mars_location import classify_lesion, lesions_of


def _seg():
    # 20x20x20: a lateral-ventricle slab, a cortex slab, WM blocks next to each, a cerebellum corner
    seg = np.zeros((20, 20, 20), np.int32)
    seg[:, :, 10] = 43                       # lateral ventricle (label 43)
    seg[:, :, 0] = 3                         # cortex
    seg[10:14, :, 11:15] = 2                 # WM block next to the ventricle
    seg[10:14, :, 1:4] = 2                   # WM block next to the cortex
    seg[:, :4, 16:20] = 7                    # cerebellum corner
    return seg


def _at(i, j, k):
    m = np.zeros((20, 20, 20), bool)
    m[i, j, k] = True
    return m


def test_wm_distance_rule():
    seg = _seg()
    deep = classify_lesion(seg, _at(11, 10, 12), (1, 1, 1))
    assert deep["mars"] == "deep", deep
    lobar = classify_lesion(seg, _at(11, 10, 2), (1, 1, 1))
    assert lobar["mars"] == "lobar" and lobar["band5mm_lobar"], lobar
    assert lobar["dist_cortex_mm"] < lobar["dist_vent_mm"]


def test_majority_labels():
    seg = _seg()
    cereb = classify_lesion(seg, _at(1, 2, 18), (1, 1, 1))
    assert cereb["mars"] == "infratentorial" and cereb["sub"] == "cerebellum", cereb
    bs = np.zeros((20, 20, 20), np.int32); bs[:] = 16
    r = classify_lesion(bs, _at(5, 5, 5), (1, 1, 1))
    assert r["mars"] == "infratentorial" and r["sub"] == "brainstem", r
    th = np.zeros((20, 20, 20), np.int32); th[:] = 11          # thalamus
    r = classify_lesion(th, _at(5, 5, 5), (1, 1, 1))
    assert r["mars"] == "deep", r
    hip = np.zeros((20, 20, 20), np.int32); hip[:] = 17        # hippocampus
    r = classify_lesion(hip, _at(5, 5, 5), (1, 1, 1))
    assert r["mars"] == "other", r
    csf = np.zeros((20, 20, 20), np.int32); csf[:] = 24        # free CSF
    r = classify_lesion(csf, _at(5, 5, 5), (1, 1, 1))
    assert r["mars"] == "other", r


def test_band_sensitivity_only_for_lobar_wm():
    seg = _seg()
    # WM voxel closer to the ventricle than the cortex, but within 5 mm of the cortex (1-voxel spacing)
    r = classify_lesion(seg, _at(11, 10, 12), (1, 1, 1))
    assert r["mars"] == "deep"
    assert isinstance(r["band5mm_lobar"], bool)


def test_lesions_of_26():
    m = np.zeros((4, 4, 4), bool)
    m[0, 0, 0] = True
    m[1, 1, 1] = True                                   # diagonal: one component in 26-neighbourhood
    m[3, 3, 3] = True
    labels, n = lesions_of(m)
    assert n == 2, n


def test_majority_counts_voxels_not_unique_labels():
    # 9-voxel lesion: 8 brainstem voxels, 1 cortex voxel -> infratentorial (11.11.2026 fix)
    import numpy as np
    from cmb.analysis.mars_location import classify_lesion
    seg = np.full((10, 10, 10), 16, np.int32)
    seg[4, 4, 5] = 3
    m = np.zeros(seg.shape, bool)
    m[4:7, 4:7, 5] = True
    assert classify_lesion(seg, m, (1, 1, 1))["mars"] == "infratentorial"
