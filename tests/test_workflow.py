"""One tiny end-to-end run per command on synthetic phantoms (CPU): preprocess -> train -> evaluate -> fine_tune
(full, norm/output only, calibration only) -> cross_validate (+ stage 2 joint and separate)."""

import glob
import json
import os

import nibabel as nib
import numpy as np

from conftest import cmb
from cmb.nifti import load


def test_preprocess_writes_the_paper_grid(grid, cohort):
    man = json.load(open(os.path.join(grid, "manifest.json")))
    assert sorted(e["id"] for e in man) == sorted(cohort["ids"])
    assert {e["fold"] for e in man} == {0, 1}
    for e in man:
        img = load(f"{grid}/{e['id']}/{e['id']}_image.nii.gz")
        assert np.allclose(img.header.get_zooms()[:3], (0.5, 0.5, 1.0))
        for kind in ("frst", "label"):
            assert load(f"{grid}/{e['id']}/{e['id']}_{kind}.nii.gz").shape == img.shape
    assert sum(e["n_cmb"] for e in man) > 0


def test_train_writes_a_loadable_model(model):
    meta = json.load(open(os.path.join(model, "seed42", "meta.json")))
    assert meta["ok"] and meta["epochs"] == 1 and meta["net"] == "a03-aniso"
    assert os.path.getsize(os.path.join(model, "seed42", "modell.pt")) > 1e7


def test_evaluate(grid, model, tmp_path):
    cmb("evaluate", "-i", grid, "-m", model, "-o", tmp_path / "pred", "--device", "cpu")
    maps = glob.glob(str(tmp_path / "pred" / "fold0" / "*_pred_proba.nii.gz"))
    assert len(maps) == 6 and len(glob.glob(str(tmp_path / "pred" / "masks" / "*_cmb_mask.nii.gz"))) == 6
    s = json.load(open(tmp_path / "pred" / "summary.json"))
    assert s["cell"] == dict(threshold=0.3, min_mm3=2.0, max_voxels=2100) and "f1" in s


def test_fine_tune_full_and_norm_out(grid, model, tmp_path):
    for layers in ("all", "norm_out"):
        out = tmp_path / layers
        cmb(
            "fine_tune",
            "-i",
            grid,
            "-m",
            model,
            "-o",
            out,
            "--ft_layers",
            layers,
            "-ep",
            "1",
            "--patches",
            "8",
            "--device",
            "cpu",
        )
        meta = json.load(open(out / "seed42" / "meta.json"))
        assert meta["variant"] == ("FTn" if layers == "norm_out" else "FT") and meta["lr"] == 1e-4
        if layers == "norm_out":
            assert meta["n_trainable_params"] < 0.01 * 22.5e6  # only norm affine + output head
        else:
            assert meta["n_trainable_params"] > 2e7


def test_fine_tune_defaults_are_the_paper_recipe():
    from cmb.cli import build_parser

    a = build_parser().parse_args(["fine_tune", "-i", "x", "-m", "pre", "-o", "y"])
    assert a.ep is None and a.ilr is None and a.ft_layers == "all"  # -> 30 epochs, lr 1e-4 in commands.fine_tune
    import inspect
    from cmb import commands

    src = inspect.getsource(commands.fine_tune)
    assert "a.ep or 30" in src and "a.ilr or 1e-4" in src and "weight_decay=1e-4" in src


def test_calibrate_only(grid, model, tmp_path):
    out = tmp_path / "cal"
    cmb("fine_tune", "-i", grid, "-m", model, "-o", out, "--calibrate_only", "True", "--device", "cpu")
    cal = json.load(open(out / "calibration.json"))
    info = json.load(open(out / "model.json"))
    assert info["cell"] == cal["cell"] and 0.05 <= cal["cell"]["threshold"] <= 0.95
    assert os.path.exists(out / "seed42" / "modell.pt")


def test_cross_validate_with_joint_and_separate_stage2(cohort, tmp_path):
    """Stage 1 by the paper's CV runner (5 folds, inner fold k+1), then stage 2 on the out-of-fold candidates. A 1-epoch
    U-Net proposes nothing useful, so before the stage-2 run the out-of-fold maps are replaced by maps with a blob on
    every reference CMB and on two random spots (finished folds are skipped by code/turnier.py): this tests the stage-2
    wiring (candidates -> joint and separate TrackSet classifiers), not accuracy."""
    g = tmp_path / "g"
    cmb(
        "preprocess",
        "-i",
        cohort["raw"],
        "-ir",
        "_T2starw",
        "-f",
        "True",
        "-b",
        cohort["raw"],
        "-br",
        "_brainmask",
        "-l",
        cohort["raw"],
        "-lr",
        "_cmb",
        "-o",
        g,
        "-fold",
        "5",
    )
    man = json.load(open(g / "manifest.json"))
    for k, e in enumerate(man):
        e["cohort"] = "siteA" if k % 2 == 0 else "siteB"
    json.dump(man, open(g / "manifest.json", "w"))
    out = tmp_path / "cv"
    cmb("cross_validate", "-i", g, "-o", out, "-ep", "1", "--patches", "8", "--device", "cpu")
    s = json.load(open(out / "summary.json"))
    assert "cv" in s["runs"]
    maps = glob.glob(str(out / "*" / "fold*" / "*_pred_proba.nii.gz"))
    assert len(maps) == 6
    from scipy import ndimage

    rng = np.random.RandomState(0)
    for path in maps:
        case = os.path.basename(path)[: -len("_pred_proba.nii.gz")]
        ni = load(path)
        lab = np.asarray(load(g / case / f"{case}_label.nii.gz").dataobj) > 0
        brain = np.asarray(load(g / case / f"{case}_image.nii.gz").dataobj) > 0
        p = 0.9 * ndimage.binary_dilation(lab, iterations=1)
        for c in np.argwhere(brain)[rng.randint(brain.sum(), size=2)]:
            p[tuple(slice(max(v - 2, 0), v + 3) for v in c)] = 0.5
        nib.save(nib.Nifti1Image(np.round(p * 255).astype(np.uint8), ni.affine), path)
    r = cmb(
        "cross_validate", "-i", g, "-o", out, "-ep", "1", "--patches", "8", "--stage2", "separate", "--device", "cpu"
    )
    text = json.dumps(json.load(open(out / "stage2.json")))
    assert "pooled tset-plain" in text and "own tset-plain" in text, r.stdout[-2000:]


def test_cross_validate_needs_five_folds(grid, tmp_path):
    r = cmb("cross_validate", "-i", grid, "-o", tmp_path, "-fold", "2", check=False)
    assert r.returncode != 0 and "5 folds" in (r.stdout + r.stderr)
