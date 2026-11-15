"""Optional stage 2 of cmb predict (after paper-v3-2026-11): the candidates are built exactly as cmb.stage2.candidates
builds them, and the classifier bundle format of cmb.stage2.final_classifier is read and applied."""
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np

from conftest import ENV, REPO, cmb


def test_candidates_equal_the_paper_module(grid, tmp_path):
    from cmb.predict_cli import stage2_candidates
    man = json.load(open(os.path.join(grid, "manifest.json")))
    case = man[1]["id"]
    d = os.path.join(grid, case)
    img = np.asarray(nib.load(f"{d}/{case}_image.nii.gz").dataobj, dtype=np.float32)
    fr = np.asarray(nib.load(f"{d}/{case}_frst.nii.gz").dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(f"{d}/{case}_label.nii.gz").dataobj) > 0
    rng = np.random.RandomState(0)
    p = np.where(lab, 0.8, 0.0) + (rng.rand(*img.shape) > 0.9995) * 0.4          # reference blobs + scattered noise
    from scipy import ndimage
    p = ndimage.grey_dilation(p, size=(2, 2, 2))
    p8 = np.round(np.clip(p, 0, 1) * 255).astype(np.uint8)
    pred = tmp_path / "pred" / "fold0"; pred.mkdir(parents=True)
    nib.save(nib.Nifti1Image(p8, nib.load(f"{d}/{case}_image.nii.gz").affine), pred / f"{case}_pred_proba.nii.gz")
    m = tmp_path / "m.json"; json.dump([dict(id=case, fold=0, dir=d)], open(m, "w"))
    r = subprocess.run([sys.executable, "-m", "cmb.stage2.candidates", "--predictions", str(tmp_path / "pred"), "--manifest", str(m),
                        "--out", str(tmp_path / "cand"), "--keep-csf", "--synthseg", "none"], cwd=REPO, env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    ref = np.load(tmp_path / "cand" / "patches.npy")
    _, ids, patches = stage2_candidates(p8.astype(np.float32) / 255.0, img, fr, dict(stored_patch=[20, 40, 40]))
    assert len(ids) == len(ref) > 0 and np.array_equal(patches, ref)


def test_bundle_is_applied(tmp_path):
    import torch
    import torch.nn as nn
    from cmb.stage2.final_classifier import build_net
    from cmb.predict_cli import load_stage2, stage2_scores
    torch.manual_seed(0)
    bundle = dict(members=[build_net(nn).state_dict() for _ in range(2)], threshold=0.5, crop=[16, 32, 32],
                  stored_patch=[20, 40, 40], candidate_threshold=0.15, candidate_min_mm3=1.0, trained_on=["test"])
    torch.save(bundle, tmp_path / "classifier.pt")
    b, nets = load_stage2(str(tmp_path / "classifier.pt"))
    s = stage2_scores(np.random.RandomState(0).rand(5, 3, 20, 40, 40).astype(np.float16), b, nets)
    assert s.shape == (5,) and np.all((s > 0) & (s < 1))
    assert np.array_equal(s, stage2_scores(np.random.RandomState(0).rand(5, 3, 20, 40, 40).astype(np.float16), b, nets))


def test_predict_with_own_stage2(cohort, model, tmp_path):
    import torch
    import torch.nn as nn
    from cmb.stage2.final_classifier import build_net
    torch.manual_seed(0)
    torch.save(dict(members=[build_net(nn).state_dict()], threshold=0.0, crop=[16, 32, 32], stored_patch=[20, 40, 40],
                    candidate_threshold=0.15, candidate_min_mm3=1.0, trained_on=["test"]), tmp_path / "classifier.pt")
    scan = os.path.join(cohort["raw"], "syn-001_T2starw.nii.gz")
    r = cmb("predict", "-i", scan, "-o", tmp_path / "out", "-m", model, "--device", "cpu", "--no-bias-correction",
            "--stage2", tmp_path / "classifier.pt")
    assert "stage 2:" in r.stdout
    import csv
    rows = list(csv.DictReader(open(tmp_path / "out" / "syn-001_T2starw_cmb.csv")))
    assert all(row["stage2_score"] != "" for row in rows)
