"""Run-stopping input checks (cmb.checks): each stops the run with a clear message on the wrong input, through the
command line where the check sits, and lets the valid input through unchanged."""

import json
import os

import nibabel as nib
import numpy as np
import pytest

from cmb import checks, synthetic
from cmb.commands import grid_manifest, score_maps
from conftest import cmb


def _img(data, aff):
    return nib.Nifti1Image(np.asarray(data), aff)


def _raw_case(
    raw, case="x01", flip_label=False, flip_mask=False, shift_mask=0.0, empty_mask=False, label_outside=False
):
    """One phantom case <case>_T2starw / _brainmask / _cmb in raw/, optionally with a broken mask or label."""
    img, lab, brain, aff = synthetic.phantom(3)
    os.makedirs(raw, exist_ok=True)
    nib.save(_img(img, aff), os.path.join(raw, f"{case}_T2starw.nii.gz"))
    m_aff = aff.copy()
    if flip_mask:  # stored LAS: x mirrored against the scan
        m_aff[0, 0] = -m_aff[0, 0]
    m_aff[0, 3] += shift_mask
    mask = np.zeros_like(brain) if empty_mask else brain
    nib.save(_img(mask, m_aff), os.path.join(raw, f"{case}_brainmask.nii.gz"))
    l_aff = aff.copy()
    if flip_label:
        l_aff[0, 0] = -l_aff[0, 0]
    if label_outside:
        lab = np.zeros_like(lab)
        lab[:3, :3, :3] = 1  # a corner of the volume: outside the brain, scan = 0
    nib.save(_img(lab, l_aff), os.path.join(raw, f"{case}_cmb.nii.gz"))


def _preprocess(raw, out, check=False):
    return cmb(
        "preprocess",
        "-i",
        raw,
        "-ir",
        "_T2starw",
        "-f",
        "True",
        "-b",
        raw,
        "-br",
        "_brainmask",
        "-l",
        raw,
        "-lr",
        "_cmb",
        "-o",
        out,
        check=check,
    )


# ------------------------------------------------------------------------------------------------ same grid
def test_same_grid_accepts_header_rounding():
    aff = np.diag([0.9, 0.9, 2.5, 1.0])
    a = _img(np.zeros((8, 8, 4), np.float32), aff)
    b = _img(np.zeros((8, 8, 4), np.uint8), aff + 1e-5)
    checks.same_grid(a, b, "scan", "mask")


@pytest.mark.parametrize(
    "what, other, message",
    [
        ("matrix", lambda aff: _img(np.zeros((8, 8, 5)), aff), "does not match the scan"),
        (
            "mirrored",
            lambda aff: _img(np.zeros((8, 8, 4)), np.diag([-0.9, 0.9, 2.5, 1.0])),
            "orientation LAS but the scan is RAS",
        ),
        (
            "shifted",
            lambda aff: _img(np.zeros((8, 8, 4)), aff + np.pad(np.array([[0.0, 0, 0, 0.5]]), ((0, 3), (0, 0)))),
            "voxel grid differs",
        ),
    ],
)
def test_same_grid_stops(what, other, message):
    aff = np.diag([0.9, 0.9, 2.5, 1.0])
    with pytest.raises(SystemExit, match=message):
        checks.same_grid(_img(np.zeros((8, 8, 4)), aff), other(aff), "scan", "mask")


def test_preprocess_stops_on_a_mirrored_label(tmp_path):
    _raw_case(str(tmp_path / "raw"), flip_label=True)
    r = _preprocess(tmp_path / "raw", tmp_path / "grid")
    assert r.returncode != 0 and "x01_cmb.nii.gz: orientation LAS but the scan is RAS" in r.stderr


def test_preprocess_stops_on_a_shifted_brain_mask(tmp_path):
    _raw_case(str(tmp_path / "raw"), shift_mask=0.9)  # one voxel along x
    r = _preprocess(tmp_path / "raw", tmp_path / "grid")
    assert r.returncode != 0 and "x01_brainmask.nii.gz: voxel grid differs from the scan" in r.stderr


def test_preprocess_accepts_the_valid_case(tmp_path):
    _raw_case(str(tmp_path / "raw"))
    r = _preprocess(tmp_path / "raw", tmp_path / "grid")
    assert r.returncode == 0, r.stderr[-2000:]


def test_predict_stops_on_a_brain_mask_of_another_grid(tmp_path):
    _raw_case(str(tmp_path / "raw"), flip_mask=True)
    raw = tmp_path / "raw"
    r = cmb(
        "predict",
        "-i",
        raw / "x01_T2starw.nii.gz",
        "--brain-mask",
        raw / "x01_brainmask.nii.gz",
        "-o",
        tmp_path / "out",
        "-m",
        tmp_path / "no_model_needed",
        "--device",
        "cpu",
        check=False,
    )
    assert r.returncode != 0 and "orientation LAS but the scan is RAS" in r.stderr


# ------------------------------------------------------------------------------------------------ inside the image
def test_mask_inside_image():
    img = np.zeros((10, 10, 10), np.float32)
    img[2:8, 2:8, 2:8] = 1
    m = np.zeros(img.shape, np.uint8)
    m[2:8, 2:8, 2:8] = 1
    checks.mask_inside_image(img, m, "mask")
    with pytest.raises(SystemExit, match="the brain mask is empty"):
        checks.mask_inside_image(img, np.zeros_like(m), "mask")
    off = np.zeros_like(m)
    off[0:4, 0:4, 0:4] = 1  # 8 of 64 voxels on data
    with pytest.raises(SystemExit, match="lies where the scan is 0"):
        checks.mask_inside_image(img, off, "mask")


def test_label_inside_image():
    img = np.zeros((10, 10, 10), np.float32)
    img[2:8, 2:8, 2:8] = 1
    lab = np.zeros(img.shape, np.uint8)
    lab[3:5, 3:5, 3:5] = 1
    lab[7:9, 7:9, 7:9] = 1  # second one half inside
    checks.label_inside_image(img, lab, "label")
    lab[0, 0, 0] = 1
    with pytest.raises(SystemExit, match="1 of 3 labelled CMB lie where the scan is 0"):
        checks.label_inside_image(img, lab, "label")


def test_preprocess_stops_on_an_empty_brain_mask(tmp_path):
    _raw_case(str(tmp_path / "raw"), empty_mask=True)
    r = _preprocess(tmp_path / "raw", tmp_path / "grid")
    assert r.returncode != 0 and "x01_brainmask.nii.gz: the brain mask is empty" in r.stderr


def test_preprocess_stops_on_a_label_outside_the_scan(tmp_path):
    _raw_case(str(tmp_path / "raw"), label_outside=True)
    r = _preprocess(tmp_path / "raw", tmp_path / "grid")
    assert r.returncode != 0 and "x01_cmb.nii.gz: 1 of 1 labelled CMB lie where the scan is 0" in r.stderr


# ------------------------------------------------------------------------------------------------ manifests
def test_unique_ids():
    checks.unique_ids([dict(id="a"), dict(id="b")], "m")
    with pytest.raises(SystemExit, match="case id\\(s\\) a occur more than once"):
        checks.unique_ids([dict(id="a"), dict(id="a")], "m")
    with pytest.raises(SystemExit, match="empty id"):
        checks.unique_ids([dict(id="")], "m")


def test_evaluate_stops_on_a_manifest_with_a_repeated_id(tmp_path):
    grid = tmp_path / "grid"
    grid.mkdir()
    json.dump(
        [dict(id="a", fold=0, dir="a", n_cmb=0), dict(id="a", fold=1, dir="a", n_cmb=0)],
        open(grid / "manifest.json", "w"),
    )
    with pytest.raises(SystemExit, match="case id\\(s\\) a occur more than once"):
        grid_manifest(str(grid))
    r = cmb("evaluate", "-i", grid, "-m", tmp_path / "model", "-o", tmp_path / "pred", check=False)
    assert r.returncode != 0 and "case id(s) a occur more than once" in r.stderr


def test_all_labelled_scans_are_scored(tmp_path):
    shape, aff = (20, 20, 10), np.diag([0.5, 0.5, 1.0, 1.0])
    (tmp_path / "pred" / "fold0").mkdir(parents=True)
    entries = []
    for cid in ("a", "b"):
        d = tmp_path / "grid" / cid
        d.mkdir(parents=True)
        nib.save(_img(np.zeros(shape, np.uint8), aff), tmp_path / "pred" / "fold0" / f"{cid}_pred_proba.nii.gz")
        entries.append(dict(id=cid, dir=str(d), labelled=True))
    nib.save(_img(np.zeros(shape, np.uint8), aff), tmp_path / "grid" / "a" / "a_label.nii.gz")  # b: label missing
    with pytest.raises(SystemExit, match="b: labelled in the manifest but no label"):
        score_maps(entries, str(tmp_path / "pred"), dict(threshold=0.3, min_mm3=2.0, max_voxels=2100))
    with pytest.raises(SystemExit, match="scored 1 scans but the manifest has 2 labelled scans"):
        checks.all_labelled_scored(1, entries)


def test_grid_without_manifest_marks_scans_without_label_unlabelled(tmp_path):
    """A grid folder without manifest.json: scans without a label file are unlabelled (not scored), as before."""
    aff = np.diag([0.5, 0.5, 1.0, 1.0])
    for cid, labelled in (("a", True), ("b", False)):
        d = tmp_path / cid
        d.mkdir()
        nib.save(_img(np.zeros((4, 4, 4), np.float32), aff), d / f"{cid}_image.nii.gz")
        if labelled:
            nib.save(_img(np.zeros((4, 4, 4), np.uint8), aff), d / f"{cid}_label.nii.gz")
    entries = json.load(open(grid_manifest(str(tmp_path))))
    assert {e["id"]: e["labelled"] for e in entries} == {"a": True, "b": False}
