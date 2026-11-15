"""Calibration-only control for phase 2 (31.10.2026), plus CAL-NEG (threshold from CMB-free scans only, see NS_NEG): is fine-tuning needed, or does it suffice to re-choose the
post-processing of the unchanged VALDO model on the n annotated local scans?

For every fold k, n and draw seed (the SAME nested STRAT draws as the S/FT blocks), the threshold t and minimum size
are chosen to maximise pooled lesion F1 on the n drawn training scans of fold k, then applied to the frozen test scans
of fold k. Grid: t in 0.05..0.95 (step 0.05) x min size in {1, 2, 3, 5, 8} mm3; max size and 26-connectivity as in
cmb.transfer.evaluate; touch matching as cmb.metrics. Also reports, per fold, the test-optimal cell (upper bound of any
calibration) and the fixed cell 0.3 / 2 mm3 (= the n = 0 point).

Input: phase4/unet/<mod>/N0_valdo72/pred/all (predict_n0.py). Output: phase4/calibration_<mod>.json and the blocks
in the cache format of plot_learning_curve.py (variant "CAL"). Aggregates only.

    python calibrate_n0.py --mod t2star
"""
from __future__ import annotations
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import argparse, json, os, sys
import numpy as np
import nibabel as nib
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R
sys.path.insert(0, ROOT)
from cmb.transfer.evaluate import MAX_VOXELS, VOXEL_MM3, CONNECTIVITY_26   # noqa: E402

GRID = {"t2star": _P + "/gitter/manifest_t2star.json",
        "swi": _P + "/gitter/manifest_swi_v2.json"}
TS = np.round(np.arange(0.05, 0.951, 0.05), 2)
MINS = [1.0, 2.0, 3.0, 5.0, 8.0]
NS = [3, 5, 10, 15, 20, 30, 40]
# CAL-NEG (31.10.2026, user: "is it enough to use only healthy patients without ROI?"): the threshold is set on n CMB-free
# training scans only (no annotation needed): the LOWEST threshold (most sensitive) whose false positives per scan on those
# scans are <= FP_TARGET, minimum size fixed at 2 mm3. FP_TARGET = 1.0 ~ the VALDO model's own cross-validation level (0.9).
NS_NEG = [3, 5, 10, 15, "all"]
FP_TARGET = 1.0


def scan_table(prob, ref):
    """counts[t, m] = (tp_ref, n_ref, n_pred, hit_pred) for every cell."""
    rl, nref = ndimage.label(ref, CONNECTIVITY_26)
    out = np.zeros((len(TS), len(MINS), 4), np.int32)
    for i, t in enumerate(TS):
        lab, n = ndimage.label(prob > t, CONNECTIVITY_26)
        if n == 0:
            out[i, :, 1] = nref; continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        hits = ndimage.maximum(ref.astype(np.uint8), lab, range(1, n + 1)) > 0          # component touches a reference
        for j, mn in enumerate(MINS):
            keep_ids = [k + 1 for k, s in enumerate(sizes) if s * VOXEL_MM3 >= mn and s <= MAX_VOXELS]
            keep = np.isin(lab, keep_ids)
            tp_ref = len(set(np.unique(rl[keep])) - {0})
            out[i, j] = (tp_ref, nref, len(keep_ids), int(sum(hits[k - 1] for k in keep_ids)))
    return out


def f1(c):
    tp, nref, npred, hp = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    fn, fp = nref - tp, npred - hp
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan), tp, fp, fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mod", choices=["t2star", "swi"], required=True)
    a = ap.parse_args()
    man = json.load(open(GRID[a.mod])); by = {e["id"]: e for e in man}
    pdir = f"{HERE}/unet/{a.mod}/N0_valdo72/pred/all"
    tab = {}
    for sid, e in by.items():
        prob = np.asarray(nib.load(f"{pdir}/{sid}_pred_proba.nii.gz").dataobj).astype(np.float32) / 255.0
        ref = np.asarray(nib.load(f"{e['dir']}/{sid}_label.nii.gz").dataobj) > 0
        tab[sid] = scan_table(prob, ref)
    i03, j2 = int(np.argmin(abs(TS - 0.3))), MINS.index(2.0)
    res, blocks = dict(mod=a.mod, grid=dict(thresholds=TS.tolist(), min_mm3=MINS), folds={}), {}
    for k in range(5):
        test = [e["id"] for e in man if e["fold"] == k]
        T = sum(tab[s] for s in test)
        ft, *_ = f1(T)
        fk = dict(test_scans=len(test), fixed_cell_f1=float(ft[i03, j2]), oracle_f1=float(np.nanmax(ft)),
                  oracle_cell=[float(TS[np.unravel_index(np.nanargmax(ft), ft.shape)[0]]),
                               MINS[np.unravel_index(np.nanargmax(ft), ft.shape)[1]]], draws={})
        for n in NS + ["all"]:
            seeds = [42] if n == "all" else ([42, 1, 7] if n <= 15 else [42, 1])
            for seed in seeds:
                if n == "all":
                    cal = [e["id"] for e in man if e["fold"] != k]
                else:
                    cal = json.load(open(f"{HERE}/draws/{a.mod}/fold{k}_n{n}_STRAT_{seed}.json"))["cases"]
                fc, *_ = f1(sum(tab[s] for s in cal))
                if np.all(np.isnan(fc)):                         # no lesion and no prediction in the n scans
                    i, j = i03, j2
                else:
                    i, j = np.unravel_index(np.nanargmax(fc), fc.shape)
                _, tp, fp, fn = f1(T[i, j])
                fk["draws"][f"n{n}_s{seed}"] = dict(cell=[float(TS[i]), MINS[j]], test_f1=float(2 * tp / max(2 * tp + fp + fn, 1)))
                blocks[f"CAL_n{n}_fold{k}_s{seed}"] = dict(variant="CAL", n=str(n), seed=seed, fold=k, tp=int(tp),
                                                           fp=int(fp), fn=int(fn), scans=len(test))
        neg_pool = [e["id"] for e in man if e["fold"] != k and e["n_cmb"] == 0]
        for n in NS_NEG:
            for seed in ([42] if n == "all" else [42, 1, 7]):
                rng = np.random.RandomState(seed); order = list(neg_pool); rng.shuffle(order)
                neg = order if n == "all" else order[:n]
                if not neg:
                    continue
                C = sum(tab[s] for s in neg)[:, j2]              # (thresholds, 4) at min size 2 mm3
                fps_neg = (C[:, 2] - C[:, 3]) / len(neg)
                ok = np.where(fps_neg <= FP_TARGET)[0]
                i = int(ok[0]) if len(ok) else len(TS) - 1
                _, tp, fp, fn = f1(T[i, j2])
                fk["draws"][f"neg_n{n}_s{seed}"] = dict(cell=[float(TS[i]), 2.0], n_neg=len(neg), test_f1=float(2 * tp / max(2 * tp + fp + fn, 1)))
                blocks[f"CALNEG_n{n}_fold{k}_s{seed}"] = dict(variant="CAL-NEG", n=str(len(neg)) if n == "all" else str(n),
                                                              seed=seed, fold=k, tp=int(tp), fp=int(fp), fn=int(fn), scans=len(test))
        _, tp, fp, fn = f1(T[i03, j2])                          # n = 0: the VALDO model as is, fixed cell, this fold
        blocks[f"N0_n0_fold{k}_s0"] = dict(variant="N0", n="0", seed=0, fold=k, tp=int(tp), fp=int(fp), fn=int(fn), scans=len(test))
        res["folds"][str(k)] = fk
        print(f"fold {k}: fixed {fk['fixed_cell_f1']:.3f}  oracle {fk['oracle_f1']:.3f} at {fk['oracle_cell']}  "
              + "  ".join(f"{d}:{v['test_f1']:.3f}" for d, v in fk["draws"].items() if d.endswith("_s42")), flush=True)
    json.dump(res, open(f"{HERE}/calibration_{a.mod}.json", "w"), indent=1)
    json.dump(blocks, open(f"{HERE}/calibration_{a.mod}_blocks.json", "w"), indent=1)
    print("written", f"{HERE}/calibration_{a.mod}.json")


if __name__ == "__main__":
    main()
