"""Property tests (hypothesis) on synthetic data: rules that must hold for EVERY input, not only for the examples in
the other tests. Each one would have caught one of the errors found after paper-v3-2026-11 (CHANGES.md).

* a left-right flip of scan + label moves the detections with it (the mirrored x axis of LAS/LPS scans)
* evaluate(prediction = reference) gives F1 = 1, an empty prediction gives tp = 0 (CMB-free scans dropped from F1)
* suffix matching never takes _roi / _brainmask names as scans and gives every scan its own id (all scans got the
  same name when the suffix was split wrongly)
"""

import csv
import os

import nibabel as nib
import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from cmb import synthetic
from cmb.commands import EXCLUDED_SUFFIXES, find_scans, score_maps
from cmb.nifti import load

CELL = dict(threshold=0.3, min_mm3=2.0, max_voxels=2100)
SHAPE = (24, 20, 10)
AFF = np.diag([0.5, 0.5, 1.0, 1.0])
SLOW = settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
FAST = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])

# a blob: corner (x, y, z) and edge lengths 2..3 voxels -> 8..27 voxels = 2..6.75 mm3 on the 0.5 x 0.5 x 1 mm grid,
# always inside the paper's cell (>= 2 mm3, <= 2100 voxels)
blob = st.tuples(
    st.integers(0, SHAPE[0] - 3),
    st.integers(0, SHAPE[1] - 3),
    st.integers(0, SHAPE[2] - 3),
    st.integers(2, 3),
    st.integers(2, 3),
    st.integers(2, 3),
)
scan_blobs = st.lists(st.lists(blob, max_size=4), min_size=1, max_size=4)  # per scan: its blobs


def _paint(blobs):
    v = np.zeros(SHAPE, np.uint8)
    for x, y, z, a, b, c in blobs:
        v[x : x + a, y : y + b, z : z + c] = 1
    return v


def _score(tmp, labels, probs, tag):
    """score_maps on scans with the given label and probability volumes (written in the grid layout)."""
    root = tmp / tag
    (root / "pred" / "fold0").mkdir(parents=True)
    entries = []
    for k, (lab, p) in enumerate(zip(labels, probs, strict=True)):
        cid = f"c{k}"
        d = root / "grid" / cid
        d.mkdir(parents=True)
        nib.save(nib.Nifti1Image(lab.astype(np.uint8), AFF), d / f"{cid}_label.nii.gz")
        nib.save(
            nib.Nifti1Image(np.round(p * 255).astype(np.uint8), AFF),
            root / "pred" / "fold0" / f"{cid}_pred_proba.nii.gz",
        )
        entries.append(dict(id=cid, dir=str(d), labelled=True))
    s = score_maps(entries, str(root / "pred"), CELL)
    masks = [
        np.asarray(load(root / "pred" / "masks" / f"c{k}_cmb_mask.nii.gz").dataobj) > 0 for k in range(len(labels))
    ]
    return s, masks


_run = iter(range(10**6))  # a fresh folder per example (the tmp_path fixture is shared by all examples)


@FAST
@given(refs=scan_blobs, preds=scan_blobs)
def test_flip_of_prediction_and_label_moves_the_detections(tmp_path, refs, preds):
    labels = [_paint(r) for r in refs]
    probs = [0.9 * _paint(p) for p in preds[: len(refs)]] + [np.zeros(SHAPE)] * max(0, len(refs) - len(preds))
    n = next(_run)
    s, masks = _score(tmp_path, labels, probs, f"a{n}")
    sf, masks_f = _score(tmp_path, [x[::-1] for x in labels], [p[::-1] for p in probs], f"b{n}")
    for k in ("tp", "fp", "fn", "f1"):
        assert s.get(k) == sf.get(k), k
    for m, mf in zip(masks, masks_f, strict=True):
        assert np.array_equal(m[::-1], mf)


@FAST
@given(refs=scan_blobs)
def test_prediction_equal_to_the_reference_gives_f1_one(tmp_path, refs):
    labels = [_paint(r) for r in refs]
    assume(any(x.any() for x in labels))
    s, _ = _score(tmp_path, labels, [0.9 * x for x in labels], f"e{next(_run)}")
    assert s["f1"] == 1.0 and s["fp"] == 0 and s["fn"] == 0
    assert s["n_scored"] == len(refs)  # CMB-free scans are scored too


@FAST
@given(refs=scan_blobs)
def test_empty_prediction_gives_no_true_positive(tmp_path, refs):
    labels = [_paint(r) for r in refs]
    s, masks = _score(tmp_path, labels, [np.zeros(SHAPE)] * len(labels), f"z{next(_run)}")
    assert s["tp"] == 0 and s["fp"] == 0 and not any(m.any() for m in masks)
    assert s["n_scored"] == len(refs)


# ------------------------------------------------------------------------------------------------ suffix matching
ident = st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_", min_size=1, max_size=10)
suffix = st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_*", min_size=1, max_size=9)


@FAST
@given(ids=st.sets(ident, min_size=1, max_size=5), suf=suffix, gz=st.booleans(), bids=st.booleans())
def test_suffix_matching_never_takes_roi_or_brainmask_as_scans(tmp_path, ids, suf, gz, bids):
    assume(not any(suf.endswith(x) for x in EXCLUDED_SUFFIXES))
    # an id must not itself end in a name that the suffix + an exclusion produces (then two files are the same name)
    assume(not any((i + suf).endswith(x) for i in ids for x in EXCLUDED_SUFFIXES))
    root = tmp_path / f"s{next(_run)}"
    ext = ".nii.gz" if gz else ".nii"
    for i in ids:
        d = root / f"sub-{i}" / "anat" if bids else root
        d.mkdir(parents=True, exist_ok=True)
        for name in (
            f"{i}{suf}",
            f"{i}{suf}_roi",
            f"{i}{suf}_brainmask",
            f"{i}_brainmask",
            f"{i}_roi",
            f"{i}{suf}_mask",
        ):
            (d / f"{name}{ext}").touch()  # find_scans looks at names only
    scans = find_scans(str(root), suf, exclude=EXCLUDED_SUFFIXES)
    for case, path in scans.items():
        stem = os.path.basename(path)[: -len(ext)]
        assert not stem.endswith(("_roi", "_brainmask", "_mask")), path
        assert stem == case + suf
    assert sorted(scans) == sorted(ids)  # every scan found once, each under its own id


# ------------------------------------------------------------------------------------------------ whole chain
def _stub_predict(models, case, image, frst, affine):
    """A stand-in for the U-Net that is NOT mirror-symmetric (a model sees left and right differently): probability
    from the image, weighted by the grid's x position. So a mirrored model input changes its output."""
    w = np.linspace(0.0, 1.0, image.shape[0])[:, None, None]  # blind on one side, as a mirrored input reveals
    return np.clip((image - 0.55) * 4 * w, 0, 1).astype(np.float32)


def _las(img, aff):
    """The same physical scan stored with the x axis mirrored (LAS): data flipped, affine adjusted."""
    flip = np.eye(4)
    flip[0, 0] = -1
    flip[0, 3] = img.shape[0] - 1
    return img[::-1].copy(), aff @ flip


@SLOW
@given(seed=st.integers(0, 10**6))
def test_left_right_flip_of_the_scan_moves_the_detections_with_it(tmp_path, monkeypatch, seed):
    """The same patient stored as RAS and as LAS must give the same CMB at the same world positions (the own profile
    reorients to RAS+ as the paper did); in voxel coordinates the detections are mirrored along x."""
    from cmb import predict_cli as P

    monkeypatch.setattr(P, "predict_grid", _stub_predict)
    img, lab, brain, aff = synthetic.phantom(seed, n_cmb=4)
    root = tmp_path / f"f{next(_run)}"
    rows = {}
    for tag, (data, a) in (("ras", (img, aff)), ("las", _las(img, aff))):
        d = root / tag
        d.mkdir(parents=True)
        nib.save(nib.Nifti1Image(data, a), d / "scan.nii.gz")
        P.run_one(
            str(d / "scan.nii.gz"), str(d / "out"), "", models=None, bias="none", log=lambda m: None, profile="own"
        )
        rows[tag] = list(csv.DictReader(open(d / "out" / "scan_cmb.csv")))
        rows[tag + "_mask"] = np.asarray(load(d / "out" / "scan_cmb_mask.nii.gz").dataobj) > 0
    assume(len(rows["ras"]) > 0)  # the phantom has CMB on the side the stand-in sees
    assert len(rows["ras"]) == len(rows["las"])
    for r, q in zip(rows["ras"], rows["las"], strict=True):
        assert r["volume_mm3"] == q["volume_mm3"]
        for k in ("x_mm", "y_mm", "z_mm"):  # written with 2 decimals: allow one step of rounding
            assert abs(float(r[k]) - float(q[k])) < 0.011, k
        assert abs(float(r["x_vox"]) - (img.shape[0] - 1 - float(q["x_vox"]))) < 0.11
    assert np.array_equal(rows["ras_mask"][::-1], rows["las_mask"])
