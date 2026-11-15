"""cmb predict on synthetic scans with an own model (-m): file and folder input, with and without --frst; outputs on the
ORIGINAL scan grid; CSV columns and size classes."""
import csv
import os

import nibabel as nib
import numpy as np

from conftest import cmb


def _check_outputs(out, name, scan):
    ref = nib.load(scan)
    for suf in ("_cmb_mask.nii.gz", "_cmb_prob.nii.gz"):
        img = nib.load(os.path.join(out, name + suf))
        assert img.shape == ref.shape and np.allclose(img.affine, ref.affine)
    rows = list(csv.DictReader(open(os.path.join(out, name + "_cmb.csv"))))
    mask = np.asarray(nib.load(os.path.join(out, name + "_cmb_mask.nii.gz")).dataobj)
    assert set(np.unique(mask)) - {0} <= set(range(1, len(rows) + 1))
    for r in rows:
        assert r["size_class"] in ("<2 mm", "2-5 mm", ">5 mm") and float(r["volume_mm3"]) >= 2.0
    return rows


def test_predict_one_file(cohort, model, tmp_path):
    scan = os.path.join(cohort["raw"], "syn-001_T2starw.nii.gz")
    r = cmb("predict", "-i", scan, "-o", tmp_path, "-m", model, "--device", "cpu", "--no-bias-correction")
    assert "syn-001_T2starw:" in r.stdout and "CMB" in r.stdout and "own model" in r.stdout
    _check_outputs(str(tmp_path), "syn-001_T2starw", scan)


def test_predict_positional_with_own_frst(cohort, model, tmp_path):
    scan = os.path.join(cohort["raw"], "syn-002_T2starw.nii.gz")
    frst = tmp_path / "frst.nii.gz"
    img = nib.load(scan)
    nib.save(nib.Nifti1Image(np.random.RandomState(0).rand(*img.shape).astype(np.float32) * 0.01, img.affine), frst)
    r = cmb("predict", scan, "-frst", frst, "-o", tmp_path / "o", "-m", model, "--device", "cpu", "--no-bias-correction")
    assert "--frst" in r.stderr                                       # the warning is printed
    _check_outputs(str(tmp_path / "o"), "syn-002_T2starw", scan)


def test_predict_folder(cohort, model, tmp_path):
    d = tmp_path / "scans"; d.mkdir()
    for k in (1, 3):
        os.symlink(os.path.join(cohort["raw"], f"syn-00{k}_T2starw.nii.gz"), d / f"syn-00{k}_T2starw.nii.gz")
    r = cmb("predict", "-i", d, "-o", tmp_path / "out", "-m", model, "--device", "cpu")
    assert "2 scans" in r.stdout
    for k in (1, 3):
        _check_outputs(str(tmp_path / "out"), f"syn-00{k}_T2starw", str(d / f"syn-00{k}_T2starw.nii.gz"))
