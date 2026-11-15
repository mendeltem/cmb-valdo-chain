"""The released weights on public VALDO data (skipped without VALDO or without network/cache access):
download + SHA-256, load, predict one VALDO case on the CPU with and without --frst, as a file and as a folder."""
import csv
import glob
import os

import pytest

from conftest import cmb, REPO

VALDO = os.environ.get("CMB_VALDO", os.path.join(REPO, "data", "valdo2021", "Task2"))
CASE = "sub-101"
SCAN = os.path.join(VALDO, CASE, f"{CASE}_space-T2S_desc-masked_T2S.nii.gz")
pytestmark = pytest.mark.skipif(not os.path.exists(SCAN), reason="VALDO not available (set CMB_VALDO)")


@pytest.fixture(scope="module")
def weights():
    r = cmb("get-model", "valdo_t2", check=False)
    if r.returncode != 0:
        pytest.skip(f"weights not downloadable here: {r.stderr[-300:]}")
    folder = r.stdout.strip().splitlines()[0]
    assert len(glob.glob(os.path.join(folder, "*", "modell.pt"))) == 2
    return folder


def test_weights_load_and_predict_a_valdo_case(weights, tmp_path):
    r = cmb("predict", "-i", SCAN, "-o", tmp_path, "--device", "cpu")
    assert "model: VALDO T2* (T2*-GRE scans only)" in r.stdout
    rows = list(csv.DictReader(open(tmp_path / f"{CASE}_space-T2S_desc-masked_T2S_cmb.csv")))
    assert len(rows) >= 1                                   # sub-101 has reference CMB (the model was trained on it)


def test_predict_folder_with_own_frst(weights, tmp_path):
    import nibabel as nib
    import numpy as np
    d = tmp_path / "in"; d.mkdir()
    os.symlink(SCAN, d / "scan.nii.gz")
    cmb("predict", "-i", d, "-o", tmp_path / "out", "--device", "cpu")
    assert (tmp_path / "out" / "scan_cmb_mask.nii.gz").exists()
    img = nib.load(SCAN)
    nib.save(nib.Nifti1Image(np.zeros(img.shape, np.float32), img.affine), tmp_path / "frst.nii.gz")
    cmb("predict", SCAN, "--frst", tmp_path / "frst.nii.gz", "-o", tmp_path / "out2", "--device", "cpu")
    assert (tmp_path / "out2" / f"{CASE}_space-T2S_desc-masked_T2S_cmb_prob.nii.gz").exists()
