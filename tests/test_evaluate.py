"""cmb evaluate scores every labelled scan as the paper did (cmb.transfer.evaluate): false positives on a CMB-free
scan count (fixed after paper-v3-2026-11, where CMB-free scans were left out of the F1)."""

import nibabel as nib
import numpy as np

from cmb.commands import score_maps


def _case(tmp, cid, ref_ball, fp_ball):
    shape = (40, 40, 20)
    aff = np.diag([0.5, 0.5, 1.0, 1.0])
    d = tmp / "grid" / cid
    d.mkdir(parents=True)
    lab = np.zeros(shape, np.uint8)
    p = np.zeros(shape, np.float32)
    if ref_ball:
        lab[10:13, 10:13, 5:8] = 1
        p[10:13, 10:13, 5:8] = 0.9  # found
    if fp_ball:
        p[25:28, 25:28, 12:15] = 0.9  # 27 voxels = 6.75 mm3: a false positive
    nib.save(nib.Nifti1Image(lab, aff), d / f"{cid}_label.nii.gz")
    (tmp / "pred" / "fold0").mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.round(p * 255).astype(np.uint8), aff), tmp / "pred" / "fold0" / f"{cid}_pred_proba.nii.gz"
    )
    return dict(id=cid, dir=str(d), n_cmb=int(ref_ball), labelled=True)


def test_false_positive_on_a_cmb_free_scan_counts(tmp_path):
    entries = [_case(tmp_path, "withcmb", True, False), _case(tmp_path, "cmbfree", False, True)]
    s = score_maps(entries, str(tmp_path / "pred"), dict(threshold=0.3, min_mm3=2.0, max_voxels=2100))
    assert s["n_scored"] == 2 and s["n_cmb_free_scored"] == 1
    assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 0
    assert abs(s["f1"] - 2 / 3) < 1e-3 and abs(s["fp_per_case"] - 0.5) < 1e-3


def test_unlabelled_scans_are_not_scored(tmp_path):
    e = _case(tmp_path, "nolabel", False, True)
    e["labelled"] = False
    s = score_maps([e], str(tmp_path / "pred"), dict(threshold=0.3, min_mm3=2.0, max_voxels=2100))
    assert s["n_scored"] == 0 and "f1" not in s
