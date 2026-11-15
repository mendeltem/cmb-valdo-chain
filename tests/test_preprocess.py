"""cmb preprocess after paper-v3-2026-11: scan matching (suffix at the end of the name, recursive, BIDS) and the grid
profiles (own = the paper's own-cohort pipeline, cmb.transfer.build_private_grids; valdo = the VALDO grid dev/gitter_d)."""

import json
import os
import shutil

import nibabel as nib
import numpy as np
import pytest

from conftest import cmb, REPO
from cmb import synthetic
from cmb.commands import find_scans, find
from cmb.nifti import load


def _write(path, data, aff):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(data, aff), path)


def test_suffix_matches_the_end_of_the_name_only(tmp_path):
    img, lab, brain, aff = synthetic.phantom(1)
    for name in ("a_T2", "a_T2_roi", "a_T2_brainmask", "b_T2", "c_T2star"):
        _write(str(tmp_path / f"{name}.nii.gz"), img, aff)
    scans = find_scans(str(tmp_path), "_T2", exclude=["_T2_roi", "_T2_brainmask"])
    assert sorted(scans) == ["a", "b"]  # not a_T2_roi, a_T2_brainmask, c_T2star
    assert str(find(str(tmp_path), "a", "_T2_roi.nii.gz")).endswith("a_T2_roi.nii.gz")


def test_bids_tree_is_searched_recursively(tmp_path):
    img, lab, brain, aff = synthetic.phantom(2)
    root = tmp_path / "bids"
    for s in ("sub-001", "sub-002"):
        _write(str(root / s / "anat" / f"{s}_T2starw.nii.gz"), img, aff)
        _write(str(root / "derivatives" / "brainmask" / s / "anat" / f"{s}_T2starw_desc-brain_mask.nii.gz"), brain, aff)
        _write(str(root / "derivatives" / "manual" / s / "anat" / f"{s}_T2starw_cmb.nii.gz"), lab, aff)
        _write(str(root / "derivatives" / "biascorr" / s / "anat" / f"{s}_desc-biascorr_T2starw.nii.gz"), img, aff)
    out = tmp_path / "grid"
    cmb(
        "preprocess",
        "-i",
        root,
        "-ir",
        "_T2starw",
        "-f",
        "True",
        "-b",
        root / "derivatives" / "brainmask",
        "-br",
        "_T2starw_desc-brain_mask",
        "-l",
        root / "derivatives" / "manual",
        "-lr",
        "_T2starw_cmb",
        "-o",
        out,
    )
    man = json.load(open(out / "manifest.json"))
    assert sorted(e["id"] for e in man) == ["sub-001", "sub-002"] and all(e["labelled"] for e in man)


def test_own_profile_equals_the_papers_own_cohort_builder(tmp_path, monkeypatch):
    """A scan in LAS orientation (x flipped): the own profile must give the grid that the paper's in-house pipeline
    (cmb.transfer.build_private_grids.prepare_source + code/gitter_c_bauen.py) gives, bit for bit."""
    img, lab, brain, aff = synthetic.phantom(3)
    aff = aff.copy()
    aff[0, 0] = -aff[0, 0]  # LAS
    raw = tmp_path / "raw"
    _write(str(raw / "x01_T2starw.nii.gz"), img, aff)
    _write(str(raw / "x01_brainmask.nii.gz"), brain, aff)
    _write(str(raw / "x01_cmb.nii.gz"), lab, aff)
    ours = tmp_path / "ours"
    cmb(
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
        ours,
        "--profile",
        "own",
    )
    import sys

    sys.path.insert(0, os.path.join(REPO, "code"))
    from cmb.transfer import build_private_grids as B
    import gitter_c_bauen as builder

    src, ref = tmp_path / "src", tmp_path / "ref"
    entry = dict(
        id="x01",
        bild=str(raw / "x01_T2starw.nii.gz"),
        hirn=str(raw / "x01_brainmask.nii.gz"),
        maske=str(raw / "x01_cmb.nii.gz"),
    )
    case = B.prepare_source(entry, "t2star", str(src))
    monkeypatch.setattr(builder, "QUELLE", str(src))
    monkeypatch.setattr(builder, "ZIEL", str(ref))
    info = builder.fall((case, 0, False))
    assert "fehler" not in info
    for kind in ("image", "frst", "label"):
        a = load(ours / "x01" / f"x01_{kind}.nii.gz")
        b = load(ref / case / f"{case}_{kind}.nii.gz")
        assert np.allclose(np.asarray(a.affine), np.asarray(b.affine)) and np.array_equal(
            np.asarray(a.dataobj), np.asarray(b.dataobj)
        ), kind
    assert nib.aff2axcodes(load(ours / "x01" / "x01_image.nii.gz").affine)[0] == "R"


VALDO = os.environ.get("CMB_VALDO", os.path.join(REPO, "data", "valdo2021", "Task2"))
REF = os.environ.get("CMB_REFERENCE_GRID", "")  # the study's dev/gitter_d (not public; set it to compare)


HAVE_FAST = bool(shutil.which("fast")) or os.path.exists(os.path.join(os.environ.get("FSLDIR", "/none"), "bin", "fast"))


@pytest.mark.valdo
@pytest.mark.skipif(
    not (os.path.isdir(VALDO) and REF and os.path.isdir(REF) and HAVE_FAST),
    reason="needs VALDO, FSL fast and the stored grid (CMB_REFERENCE_GRID)",
)
def test_valdo_profile_reproduces_the_stored_valdo_grid(tmp_path):
    case = "sub-101"
    raw = tmp_path / "raw"
    raw.mkdir()
    os.symlink(os.path.join(VALDO, case, f"{case}_space-T2S_desc-masked_T2S.nii.gz"), raw / f"{case}_T2S.nii.gz")
    out = tmp_path / "grid"
    cmb("preprocess", "-i", raw, "-ir", "_T2S", "-o", out, "--profile", "valdo")
    for kind in ("image", "frst"):
        a = np.asarray(load(out / case / f"{case}_{kind}.nii.gz").dataobj)
        b = np.asarray(load(os.path.join(REF, case, f"{case}_{kind}.nii.gz")).dataobj)
        assert a.shape == b.shape and np.array_equal(a, b), kind
