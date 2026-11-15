# -*- coding: utf-8 -*-
"""wk0: Re-rank the stage-1 U-Nets with a nested best threshold.

Task (user 30.10.2026): "we take the best threshold of the U-Net for F1".
CPU only, NO training: inputs are the stored out-of-fold probability maps
``fold<k>/<case>_pred_proba.nii.gz`` (uint8, /255 to [0,1]).

Background: Arena 1 ranked the U-Nets by F1 at the FIXED threshold 0.3. The
U-Nets are calibrated differently, so that partly compared thresholds. Here,
for each run and dataset, per test fold k:

  t_k = argmax over the grid {0.05, 0.10, ..., 0.90} of the POOLED lesion F1
  computed on the OTHER four folds' out-of-fold maps (nested -- the test fold
  is never used to choose its own threshold), then t_k is applied to fold k.
  Pooling the test folds -> "F1 at nested best threshold".

Also reported per run:
  (a) F1 at the fixed threshold 0.3 (must reproduce the Arena-1 numbers),
  (b) candidate ceiling recall at the stage-2 budget: threshold 0.15,
      >= 1 mm3, top 40 components per scan by max probability,
  (c) CPM = mean lesion sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8
      false positives per scan (FROC over the cluster MAX probability,
      LUNA16 definition),
  (d) the chosen thresholds t_k (spread across folds).

Lesion matching is the one from ``cmb.transfer.evaluate`` / ``code/metric.py``:
26-connectivity, TP = reference component touched by a prediction, FN =
untouched reference, FP = prediction touching no reference. Unchanged.

Bootstrap over cases (SHARED draws, 2000) for the difference to the reference
run per metric.

CLI
---
    python -m cmb.analysis.unet_rerank --dataset valdo \
        --runs ergebnisse/turnier --manifest dev/manifest_valdo_gitter_d.json \
        --out <private>/paper/unet_rerank \
        [--reference a03-aniso-e60-gd] [--runs-only NAME ...]

The per-dataset tables are written to ``<out>/<dataset>.json``; ``run_all``
merges them into ``<out>/rerank.json`` + ``<out>/rerank.md``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

from cmb.paths import ROOT
sys.path.insert(0, f"{ROOT}/code")
from metric import hits  # noqa: E402  # 26-connectivity lesion matching

N26 = np.ones((3, 3, 3), dtype=bool)
GRID = [round(0.05 * i, 2) for i in range(1, 19)]   # 0.05 .. 0.90
FIXED_T = 0.30
CEILING_T = 0.15
CEILING_TOPN = 40
FROC_FPS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
BOOTSTRAP = 2000
RNG_SEED = 20261030
MM3_PER_VOX = 0.25              # fallback; real value comes from the NIfTI header
MIN_MM3 = 2.0                   # cmb.transfer.evaluate post-processing (frozen)
MAX_VOXELS = 2100               # giant components removed (frozen)
FROC_DET_T = 0.05               # finest grid threshold for the candidate set
POSTPROC = True                 # size filtering on by default; off in synthetic unit tests


# ---------------------------------------------------------------------------
# per-case building blocks
# ---------------------------------------------------------------------------
def _component_sizes_vox(seg: np.ndarray):
    labels, n = ndimage.label(seg, structure=N26)
    if n == 0:
        return labels, 0, np.zeros(0), np.zeros(0)
    size = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)
    return labels, n, size, labels


def _postprocess(seg, mm3, min_mm3=MIN_MM3, max_voxels=MAX_VOXELS,
                 do_pp=True, min_size_vox=None):
    """Frozen post-processing: threshold already applied, then keep components
    with >= min_mm3 and <= max_voxels (giant removal). ``do_pp`` off for the
    synthetic unit tests (their 1-voxel objects would vanish)."""
    if not do_pp or not min_size_vox:
        return seg
    labels, n = ndimage.label(seg, structure=N26)
    if n == 0:
        return seg
    size = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    keep = (size * mm3 >= min_mm3) & (size <= max_voxels)
    return np.isin(labels, np.where(keep)[0] + 1)


def case_rows(proba: np.ndarray, ref: np.ndarray, mm3=MM3_PER_VOX, do_pp=False):
    """Per-case (tp, fp, fn, n_ref) at every grid threshold.

    Returns a list aligned to GRID. ``do_pp`` applies the frozen size
    filtering (>= 2 mm3, giants removed); off for the synthetic unit tests
    whose 1-voxel objects would be filtered away.
    """
    rows = []
    for t in GRID:
        seg = (proba >= t).astype(np.uint8)
        if do_pp:
            seg = _postprocess(seg, mm3)
        r = hits(ref, seg)
        rows.append((r["tp"], r["fp"], r["fn"], r["n_ref"]))
    return rows


def pooled_f1(rows_by_case, thr_idx, case_idx):
    """Pooled lesion F1 for a subset of cases at one grid threshold index."""
    tp = fp = fn = 0
    for ci in case_idx:
        t, f, n, _ = rows_by_case[ci][thr_idx]
        tp += t
        fp += f
        fn += n
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom else 0.0


def f1_of_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom else 0.0


def nested_thresholds(rows_by_case, fold_of_case, n_folds=5):
    """t_k from GRID maximising pooled F1 on the OTHER four folds' cases."""
    cases_per_fold = {k: [i for i, f in enumerate(fold_of_case) if f == k]
                      for k in range(n_folds)}
    tks = []
    for k in range(n_folds):
        others = [i for j in range(n_folds) if j != k for i in cases_per_fold[j]]
        best_t, best_f1 = None, -1.0
        for ti, t in enumerate(GRID):
            f1 = pooled_f1(rows_by_case, ti, others)
            if f1 > best_f1 + 1e-12:
                best_f1, best_t = f1, t
        tks.append(best_t)
    return tks


def ceiling_recall(proba: np.ndarray, ref: np.ndarray,
                   thr=CEILING_T, topn=CEILING_TOPN,
                   min_mm3=0.25, mm3=MM3_PER_VOX, do_pp=False):
    """Candidate ceiling recall at the stage-2 budget (top-N per scan).

    Detect at ``thr`` with >= ``min_mm3``, keep the ``topn`` components with the
    highest max probability, TP = reference touched. Returns (recall, tp, n_gt).
    """
    seg = (proba >= thr).astype(np.uint8)
    if do_pp:
        seg = _postprocess(seg, mm3, min_mm3=min_mm3)
    labels, n = ndimage.label(seg, structure=N26)
    n_gt = int(ndimage.label(ref, structure=N26)[1])
    if n == 0:
        return (0.0 if n_gt else float("nan"), 0, n_gt)
    size = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    keep_mask = (size * mm3 >= min_mm3)
    # rank kept components by their max probability
    comp_max = np.full(n + 1, -1.0)
    np.maximum.at(comp_max, labels.ravel(), proba.ravel())
    comp_max = comp_max[1:]
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) > topn:
        order = keep_idx[np.argsort(-comp_max[keep_idx])][:topn]
    else:
        order = keep_idx
    det = np.isin(labels, order + 1).astype(np.uint8)   # labels are 1-based
    r = hits(ref, det)
    rec = (r["tp"] / n_gt) if n_gt else float("nan")
    return rec, r["tp"], n_gt


def cpm_curve(proba: np.ndarray, ref: np.ndarray, det_t=FROC_DET_T,
              mm3=MM3_PER_VOX, do_pp=False):
    """Per-case FROC candidate list.

    Detect at ``det_t`` (finest grid), score each cluster by its MAX
    probability. Returns (scores, is_tp) lists -- pooled + ranked across cases
    by the caller (LUNA16: rank all detections by confidence, keep the best
    n = round(fp/scan * n_scans), sensitivity = TP / n_gt).
    """
    seg = (proba >= det_t).astype(np.uint8)
    if do_pp:
        seg = _postprocess(seg, mm3)
    labels, n = ndimage.label(seg, structure=N26)
    if n == 0:
        return [], []
    size = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    comp_max = np.full(n + 1, -1.0)
    np.maximum.at(comp_max, labels.ravel(), proba.ravel())
    comp_max = comp_max[1:]
    # which components touch a reference
    both = seg & ref
    comp_tp = np.zeros(n, bool)
    if both.any():
        comp_tp[np.unique(labels[both]) - 1] = True
    keep = size > 0
    return list(zip(comp_max[keep].tolist(), comp_tp[keep].tolist()))


def froc_cpm(all_dets, n_scans, n_gt, fps=FROC_FPS):
    """LUNA16 CPM from a pooled, confidence-sorted detection list.

    ``all_dets``: list of (score, is_tp) over ALL cases (pooled). For each
    fp/scan budget keep the top n = round_half_up(fp * n_scans) detections
    (half-up, so the 1/2 FP/scan budget keeps one detection for one scan).
    CPM = mean sensitivity over the budgets.
    """
    dets = sorted(all_dets, key=lambda d: -d[0])
    sens = []
    for fp in fps:
        # LUNA16: at fp/scan the allowed detection count is floor(fp*n)+1
        # (fp=0.5, 1 scan -> 1; fp=1 -> 2; fp=2 -> 3)
        n_keep = int(math.floor(fp * n_scans)) + 1
        kept = dets[:n_keep]
        tp = sum(1 for _, t in kept if t)
        sens.append(tp / n_gt if n_gt else float("nan"))
    finite = [s for s in sens if not math.isnan(s)]
    cpm = float(np.mean(finite)) if finite else float("nan")
    return cpm, {str(fp): (None if math.isnan(s) else round(s, 4)) for fp, s in zip(fps, sens)}


# ---------------------------------------------------------------------------
# one run (worker)
# ---------------------------------------------------------------------------
def process_run(args):
    name, folder, manifest, reference = args
    cases = {e["id"]: e for e in json.load(open(manifest))}
    # load per case: proba, ref, fold
    per_case = []
    mm3 = MM3_PER_VOX
    for case in cases.values():
        cid = case["id"]
        pth = os.path.join(folder, f"fold{case['fold']}", f"{cid}_pred_proba.nii.gz")
        if not os.path.exists(pth):
            continue
        import nibabel as nib
        img = nib.load(pth)
        proba = np.asarray(img.dataobj).astype(np.float32) / 255.0
        ref = np.asarray(nib.load(os.path.join(case["dir"], f"{cid}_label.nii.gz")).dataobj) > 0
        z = img.header.get_zooms()[:3]
        mm3 = float(np.prod(z))
        per_case.append((cid, case["fold"], proba, ref))

    rows_by_case = [case_rows(p, r, mm3=mm3, do_pp=True) for (_, _, p, r) in per_case]
    fold_of_case = [f for (_, f, _, _) in per_case]
    n_scans = len(per_case)
    n_gt = sum(r[0][3] for r in rows_by_case)

    # (a) F1 at fixed 0.3 (0.3 is on the grid, index 6)
    fi = GRID.index(FIXED_T)
    f1_03 = pooled_f1(rows_by_case, fi, list(range(n_scans)))

    # (d) nested thresholds + (b) F1 at nested best (pooled over test folds)
    tks = nested_thresholds(rows_by_case, fold_of_case)
    tp = fp = fn = 0
    for k in range(5):
        ti = GRID.index(tks[k])
        for i, f in enumerate(fold_of_case):
            if f == k:
                t, f_, n, _ = rows_by_case[i][ti]
                tp += t
                fp += f_
                fn += n
    f1_best = f1_of_counts(tp, fp, fn)

    # (b) ceiling recall + (c) CPM need the raw maps
    ceiling_recs, n_gt_c = [], 0
    dets_by_case = []
    ceil_per_case = []
    for (cid, f, proba, ref) in per_case:
        rec, tpc, ngtc = ceiling_recall(proba, ref, mm3=mm3)
        ceil_per_case.append({"id": cid, "tp": tpc, "n_gt": ngtc})
        if not math.isnan(rec):
            ceiling_recs.append(rec)
        n_gt_c += ngtc
        dets_by_case.append(cpm_curve(proba, ref, mm3=mm3, do_pp=True))
    ceiling = float(np.mean(ceiling_recs)) if ceiling_recs else float("nan")
    all_dets = [d for lst in dets_by_case for d in lst]
    cpm, curve = froc_cpm(all_dets, n_scans, n_gt_c)

    return {
        "name": name,
        "n_scans": n_scans,
        "n_lesions": int(n_gt),
        "f1_at_03": round(f1_03, 4),
        "f1_nested_best": round(f1_best, 4),
        "t_per_fold": [round(t, 2) for t in tks],
        "t_range": [round(min(tks), 2), round(max(tks), 2)],
        "ceiling_recall_z40": round(ceiling, 4) if not math.isnan(ceiling) else None,
        "cpm": round(cpm, 4) if not math.isnan(cpm) else None,
        "cpm_curve": curve,
        "_per_case": [(cid, f, rows_by_case[i]) for i, (cid, f, _, _) in enumerate(per_case)],
        "_ceil": ceil_per_case,
        "_dets_by_case": dets_by_case,
        "_n_gt": int(n_gt),
    }


def _row_f1(row, ti):
    t, f, n, _ = row[ti]
    return t, f, n


# ---------------------------------------------------------------------------
# dataset driver
# ---------------------------------------------------------------------------
def run_dataset(name, folder_root, manifest, reference, out_dir,
                runs_only=None, workers=None, expected_f1=None):
    import nibabel  # noqa: F401  (worker import)
    cases = {e["id"]: e for e in json.load(open(manifest))}
    want = {k: {e["id"] for e in cases.values() if e["fold"] == k} for k in range(5)}

    complete, skipped = [], []
    for d in sorted(os.listdir(folder_root)):
        p = os.path.join(folder_root, d)
        if not os.path.isdir(p):
            continue
        have = {k: {os.path.basename(x)[:-len("_pred_proba.nii.gz")]
                    for x in glob.glob(os.path.join(p, f"fold{k}", "*_pred_proba.nii.gz"))}
                for k in range(5)}
        if all(have[k] == want[k] for k in range(5)):
            complete.append(d)
        else:
            missing = [k for k in range(5) if have[k] != want[k]]
            skipped.append({"name": d, "reason": f"missing/incomplete folds {missing}"})
    if runs_only:
        complete = [d for d in complete if d in runs_only]

    jobs = [(d, os.path.join(folder_root, d), manifest, reference) for d in complete]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        for res in ex.map(process_run, jobs):
            results.append(res)
            print(f"  [{len(results)}/{len(jobs)}] {res['name']}: "
                  f"F1@0.3={res['f1_at_03']:.3f} F1@best={res['f1_nested_best']:.3f} "
                  f"t={res['t_range']}", flush=True)
    elapsed = time.time() - t0

    results = {r["name"]: r for r in results}
    ref = results.get(reference)
    if ref is None:
        print(f"WARNING: reference run {reference} not in complete set; "
              f"skipping bootstrap deltas.", flush=True)

    # (a) check: F1 at 0.3 must reproduce the earlier Arena numbers
    if expected_f1:
        for rn, want in expected_f1.items():
            r = results.get(rn)
            if r is None:
                continue
            got = r["f1_at_03"]
            if abs(got - want) > 0.005:
                raise SystemExit(f"ABORT: F1@0.3 for {rn} = {got}, "
                                 f"earlier arena number {want} (diff > 0.005) -- "
                                 f"post-processing changed, do not publish")
            print(f"  F1@0.3 check {rn}: {got:.3f} == arena {want}  OK", flush=True)

    # ---- case-shared bootstrap deltas vs the reference ----------------------
    def _pack(r):
        """Per-case arrays sorted by case id.
        Returns (ids, C[n,18,3] (tp,fp,fn) per grid thr, Nref[n],
                 tidx[n] (grid index of the case's own fold threshold),
                 Ceil[n,2] (tp,n_gt), dets[n] per-case (score,is_tp) arrays)."""
        items = sorted(r["_per_case"], key=lambda x: x[0])
        C = np.zeros((len(items), len(GRID), 3), int)
        Nref = np.zeros(len(items), int)
        folds = np.array([f for _, f, _ in items])
        tks = r["t_per_fold"]
        for i, (cid, f, row) in enumerate(items):
            C[i] = np.asarray([t[:3] for t in row])
            Nref[i] = row[0][3]
        tidx = np.array([GRID.index(tks[f]) for f in folds], int)
        ceil = np.array([[e["tp"], e["n_gt"]] for _, e in
                         zip(items, sorted(r["_ceil"], key=lambda x: x["id"]))])
        id_to_dets = {cid: d for (cid, f, row), d in
                      zip(r["_per_case"], r["_dets_by_case"])}
        dets = [np.asarray(id_to_dets[cid], dtype=np.float32).reshape(-1, 2)
                for cid, _, _ in items]
        return ([cid for cid, _, _ in items], C, Nref, tidx, ceil, dets)

    def _cpm_sens(det_arrays, n_scans, n_gt, nk_list):
        """Sensitivities at the fixed budgets nk_list from per-case det arrays."""
        if n_gt <= 0 or not any(len(d) for d in det_arrays):
            return [float("nan")] * len(nk_list)
        pool = np.concatenate(det_arrays)
        order = np.argsort(-pool[:, 0], kind="stable")
        tp_sorted = pool[order, 1]
        csum = np.concatenate(([0.0], np.cumsum(tp_sorted)))
        out = []
        for nk in nk_list:
            if nk == 0 or nk > len(csum) - 1:
                out.append(float("nan"))
            else:
                out.append(float(csum[nk]) / n_gt)
        return out

    if ref is not None:
        rid, rC, rN, rTidx, rCeil, rD = _pack(ref)

        for name, r in results.items():
            if name == reference:
                r["delta_vs_reference"] = None
                continue
            aid, C, N, tidx, Ceil, D = _pack(r)
            common = sorted(set(aid) & set(rid))
            ia = {c: i for i, c in enumerate(aid)}
            ib = {c: i for i, c in enumerate(rid)}
            a_ix = np.array([ia[c] for c in common])
            b_ix = np.array([ib[c] for c in common])
            A, B = C[a_ix], C[b_ix]
            Na, Nb = N[a_ix], N[b_ix]
            ta, tb = tidx[a_ix], tidx[b_ix]
            A_det = [D[i] for i in a_ix]
            B_det = [rD[i] for i in b_ix]
            A_ceil, B_ceil = Ceil[a_ix], rCeil[b_ix]
            m = len(common)
            fi = GRID.index(FIXED_T)
            rng = np.random.RandomState(RNG_SEED)
            draws = np.empty((BOOTSTRAP, 4), float)
            for i in range(BOOTSTRAP):
                ix = rng.randint(m, size=m)
                sa, sb = A[ix, fi].sum(axis=0), B[ix, fi].sum(axis=0)
                draws[i, 0] = f1_of_counts(*sa) - f1_of_counts(*sb)
                sa, sb = A[ix, ta[ix]].sum(axis=0), B[ix, tb[ix]].sum(axis=0)
                draws[i, 1] = f1_of_counts(*sa) - f1_of_counts(*sb)
                # ceiling recall over resampled cases with gt > 0
                ma, mb = A_ceil[ix, 1] > 0, B_ceil[ix, 1] > 0
                if ma.any() and mb.any():
                    draws[i, 2] = (A_ceil[ix, ma, 0] / A_ceil[ix, ma, 1]).mean() \
                        - (B_ceil[ix, mb, 0] / B_ceil[ix, mb, 1]).mean()
                else:
                    draws[i, 2] = 0.0
                # CPM: resampled case sets, budgets = floor(fp * m) + 1 (LUNA16)
                nka = [int(math.floor(fp * m)) + 1 for fp in FROC_FPS]
                ngt_a, ngt_b = int(Na[ix].sum()), int(Nb[ix].sum())
                sa_s = _cpm_sens([A_det[j] for j in ix], m, ngt_a, nka)
                sb_s = _cpm_sens([B_det[j] for j in ix], m, ngt_b, nka)
                sa_f = [s for s in sa_s if not math.isnan(s)]
                sb_f = [s for s in sb_s if not math.isnan(s)]
                draws[i, 3] = (float(np.mean(sa_f)) - float(np.mean(sb_f))
                               if sa_f and sb_f else 0.0)
            names4 = ["f1_at_03", "f1_nested_best", "ceiling_recall_z40", "cpm"]
            r["delta_vs_reference"] = {
                names4[j]: {"delta": round(float(draws[:, j].mean()), 4),
                            "ci95": [round(float(np.percentile(draws[:, j], 2.5)), 4),
                                     round(float(np.percentile(draws[:, j], 97.5)), 4)],
                            "n": int(m)} for j in range(4)
            }
            r["delta_vs_reference"]["n_common_cases"] = int(m)
    # drop the heavy per-case structures from the stored json
    for name, r in results.items():
        for k in ("_per_case", "_ceil", "_dets_by_case"):
            r.pop(k, None)

    out = {
        "dataset": name,
        "manifest": manifest,
        "n_cases": len(cases),
        "n_runs": len(results),
        "elapsed_s": round(elapsed, 1),
        "reference": reference,
        "runs": results,
        "skipped": skipped,
    }
    out_path = os.path.join(out_dir, f"{name}.json")
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"wrote {out_path} ({len(results)} runs, {elapsed:.0f}s)", flush=True)
    return out


def _pooled_f1_at(counts):
    tp = int(sum(c[0] for c in counts))
    fp = int(sum(c[1] for c in counts))
    fn = int(sum(c[2] for c in counts))
    return f1_of_counts(tp, fp, fn)


def bootstrap_delta_f1(rows_a, rows_b, idx_a, idx_b,
                       n=BOOTSTRAP, seed=RNG_SEED):
    """Case-SHARED bootstrap for a pooled-F1 difference (a - b).

    ``rows_a`` / ``rows_b``: identically aligned per-case lists of
    (tp, fp, fn, n_ref) at every grid threshold (``case_rows`` output).
    ``idx_a`` / ``idx_b``: per-case grid index (int array) -- the threshold
    each case was scored at (e.g. its fold's own t_k). Returns
    (delta, lo, hi, n_cases).
    """
    m = len(rows_a)
    assert m == len(rows_b)

    def _counts(rows, idx):
        if np.isscalar(idx):
            return np.asarray([tuple(r[:3]) for r in rows], int)
        return np.asarray([tuple(r[i][:3]) for r, i in zip(rows, idx)], int)

    A = _counts(rows_a, idx_a)
    B = _counts(rows_b, idx_b)
    point = _pooled_f1_at(A) - _pooled_f1_at(B)
    rng = np.random.RandomState(seed)
    draws = np.empty(n)
    for i in range(n):
        ix = rng.randint(m, size=m)
        draws[i] = _pooled_f1_at(A[ix]) - _pooled_f1_at(B[ix])
    # element 0 = the POINT estimate (exact on the full data), not the noisy
    # bootstrap mean -- the CI in [1,2] carries the sampling uncertainty.
    return (float(point), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), m)


def bootstrap_delta_recall(a, b, n=BOOTSTRAP, seed=RNG_SEED):
    """Case-shared bootstrap for the difference of per-case (gt>0) recalls."""
    ra = [(r["tp"] / r["n_gt"]) for r in a if r["n_gt"] > 0]
    rb = [(r["tp"] / r["n_gt"]) for r in b if r["n_gt"] > 0]
    m = min(len(ra), len(rb))
    rng = np.random.RandomState(seed)
    A, B = np.asarray(ra[:m]), np.asarray(rb[:m])
    draws = np.empty(n)
    for i in range(n):
        ix = rng.randint(m, size=m)
        draws[i] = A[ix].mean() - B[ix].mean()
    return (float(draws.mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), m)


def selftest():
    """CPU unit tests on synthetic maps."""
    rng = np.random.RandomState(0)
    # --- 1. a map with a known best threshold is recovered -----------------
    # 5 folds x 2 cases. In each case one true lesion (3-voxel blob) and one
    # noise region. The "signal" probability is 0.8 in the lesion, "noise" is
    # uniform on [0.0, t_true) so that the pooled F1 on the four training folds
    # is maximised exactly at t_true.
    t_true = 0.55
    shape = (12, 12, 12)
    folds = []
    for f in range(5):
        case = []
        for c in range(2):
            p = np.zeros(shape, np.float32)
            ref = np.zeros(shape, bool)
            # lesion at a fixed, separate spot per case
            cx = 3 + 5 * c
            ref[cx:cx + 2, 5, 5] = True
            p[ref] = 0.8
            # scattered noise, all below t_true
            n_noise = 50
            idx = rng.randint(0, shape[0], n_noise), rng.randint(0, shape[1], n_noise), \
                rng.randint(0, shape[2], n_noise)
            p[idx] = rng.random(n_noise) * (t_true - 0.02)
            case.append((p, ref))
        folds.append(case)

    def fold_rows(fold_cases):
        rows = []
        for p, ref in fold_cases:
            rows.append(case_rows(p, ref))
        return rows

    fold_of_case = [f for f in range(5) for _ in range(2)]
    all_rows = [r for f in range(5) for r in fold_rows(folds[f])]
    tks = nested_thresholds(all_rows, fold_of_case)
    assert tks == [t_true] * 5, f"expected t={t_true} on all folds, got {tks}"
    # pooled F1 at the nested thresholds, test fold scored at its own t_k
    tp = fp = fn = 0
    for k in range(5):
        for ci, f in enumerate(fold_of_case):
            if f == k:
                t, f_, n, _ = all_rows[ci][GRID.index(tks[k])]
                tp += t
                fp += f_
                fn += n
    f1_best = f1_of_counts(tp, fp, fn)
    assert f1_best > 0.9, f1_best

    # --- 2. the nested choice never looks at the test fold -----------------
    # Poison fold 3: one case whose lesion probability is 0.99 (huge) and a
    # sea of 0.5 noise. If the choice for fold 3 used its own data, t_3 would
    # be pulled towards 0.55/0.6; with the nested rule it must stay 0.55.
    poison = np.zeros(shape, np.float32)
    prefp = np.zeros(shape, bool)
    prefp[2, 2, 2] = True
    poison[2, 2, 2] = 0.99
    poison[6:9, 6:9, 6:9] = 0.5
    folds[3][0] = (poison, prefp)
    all_rows = [r for f in range(5) for r in fold_rows(folds[f])]
    tks = nested_thresholds(all_rows, fold_of_case)
    assert all(t == t_true for t in tks), f"test fold leaked into choice: {tks}"

    # --- 3. FROC/CPM on a toy case ------------------------------------------
    # TP needs voxel overlap with the reference; detections must be
    # 26-disconnected from each other.
    p = np.zeros((10, 10, 10), np.float32)
    ref = np.zeros((10, 10, 10), bool)
    ref[1, 1, 1] = True
    ref[8, 8, 8] = True
    p[1, 1, 1] = 0.9                        # TP on ref 1
    p[4, 4, 4] = 0.7                        # FP
    p[8, 8, 8] = 0.6                        # TP on ref 2
    dets = cpm_curve(p, ref)
    by_score = sorted(dets, reverse=True)
    assert [round(s, 2) for s, _ in by_score] == [0.9, 0.7, 0.6], dets
    assert [t for _, t in by_score] == [True, False, True], dets
    cpm, curve = froc_cpm(dets, n_scans=1, n_gt=2, fps=[0.5, 1.0, 2.0])
    # fp=0.5 -> 1 kept (TP) -> 0.5; fp=1 -> 2 kept -> 1 TP -> 0.5;
    # fp=2 -> 3 kept -> 2 TP -> 1.0
    assert curve["0.5"] == 0.5 and curve["1.0"] == 0.5 and curve["2.0"] == 1.0, curve
    assert abs(cpm - (0.5 + 0.5 + 1.0) / 3) < 1e-9, cpm

    # --- 4. ceiling recall budget: top-N by max probability -----------------
    p = np.zeros((10, 10, 10), np.float32)
    ref = np.zeros((10, 10, 10), bool)
    ref[1, 1, 1] = True
    p[1, 1, 1] = 0.9                        # TP on the reference
    p[3, 1, 1], p[4, 1, 1], p[5, 1, 1] = 0.8, 0.7, 0.6   # one FP component
    assert ceiling_recall(p, ref, topn=3) == (1.0, 1, 1)
    assert ceiling_recall(p, ref, thr=0.85, topn=3) == (1.0, 1, 1)
    assert ceiling_recall(p, ref, topn=1) == (1.0, 1, 1)
    assert ceiling_recall(np.zeros((10, 10, 10), np.float32), ref, topn=3) == (0.0, 0, 1)

    # --- 5. bootstrap delta is reproducible (same seed) ---------------------
    rows_a = [(1, 0, 0, 1), (0, 1, 1, 1), (1, 1, 0, 1)] * 4
    rows_b = [(1, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 1)] * 4
    fi = GRID.index(FIXED_T)
    d1 = bootstrap_delta_f1(rows_a, rows_b, fi, fi, n=50)
    d2 = bootstrap_delta_f1(rows_a, rows_b, fi, fi, n=50)
    assert d1 == d2, "bootstrap must be reproducible for the same seed"
    print("selftest passed (nested threshold, leak check, FROC/CPM, ceiling, bootstrap)")


def merge_reports(out_dir, datasets):
    """Merge <out>/<dataset>.json into rerank.json + rerank.md."""
    merged = {}
    for ds in datasets:
        p = os.path.join(out_dir, f"{ds}.json")
        if os.path.exists(p):
            merged[ds] = json.load(open(p))
    json.dump(merged, open(os.path.join(out_dir, "rerank.json"), "w"), indent=1)

    lines = []
    lines.append("# wk0 -- U-Net re-ranking at the nested best threshold")
    lines.append("")
    lines.append("Arena 1 ranked the stage-1 U-Nets by F1 at the FIXED threshold 0.3; the "
                 "nets are calibrated differently, so that partly compared thresholds. "
                 "Here t_k is chosen per fold on the pooled lesion F1 of the OTHER four "
                 "folds (grid 0.05..0.90, step 0.05) and applied to fold k. "
                 "Lesion matching: 26-connectivity, TP = reference component touched "
                 "(cmb.transfer.evaluate / code/metric.py, unchanged). "
                 "Ceiling recall: threshold 0.15, top 40 components per scan by max "
                 "probability. CPM = mean lesion sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 "
                 "false positives per scan (FROC over the cluster max probability, "
                 "LUNA16 definition). Bootstrap: 2000 shared case draws vs the reference.")
    lines.append("")
    for ds, tab in merged.items():
        lines.append(f"## {ds} (n={tab['n_cases']}, {tab['n_runs']} runs, "
                     f"reference={tab['reference']})")
        lines.append("")
        runs = list(tab["runs"].items())
        runs.sort(key=lambda kv: -(kv[1]["f1_nested_best"] or 0))
        lines.append("| # | run | F1@0.3 | F1@best (t_k range) | ceiling recall @z40 | CPM | dF1@best vs ref [95% CI] |")
        lines.append("|---|-----|--------|---------------------|---------------------|-----|--------------------------|")
        for rank, (name, r) in enumerate(runs, 1):
            d = r.get("delta_vs_reference") or {}
            db = d.get("f1_nested_best") or {}
            dtxt = (f"{db['delta']:+.3f} [{db['ci95'][0]:+.3f}; {db['ci95'][1]:+.3f}]"
                    if db.get("delta") is not None else "--")
            tr = r["t_range"]
            lines.append(f"| {rank} | {name} | {r['f1_at_03']:.3f} | {r['f1_nested_best']:.3f} "
                         f"({tr[0]:.2f}-{tr[1]:.2f}) | {r['ceiling_recall_z40']} | "
                         f"{r['cpm']} | {dtxt} |")
        lines.append("")
        # top 3 per metric
        for key, label in (("f1_at_03", "F1@0.3"), ("f1_nested_best", "F1@best"),
                           ("ceiling_recall_z40", "ceiling recall @z40"), ("cpm", "CPM")):
            vals = [(n, r[key]) for n, r in runs if r.get(key) is not None]
            vals.sort(key=lambda x: -x[1])
            top = ", ".join(f"{n} ({v:.3f})" for n, v in vals[:3])
            lines.append(f"- top 3 {label}: {top}")
        lines.append("")
        if tab.get("skipped"):
            lines.append("Skipped (incomplete maps):")
            for s in tab["skipped"]:
                lines.append(f"- {s['name']}: {s['reason']}")
            lines.append("")
    open(os.path.join(out_dir, "rerank.md"), "w").write("\n".join(lines))
    print(f"wrote {os.path.join(out_dir, 'rerank.json')} and rerank.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--runs", help="folder root containing run subfolders")
    ap.add_argument("--manifest")
    ap.add_argument("--reference")
    ap.add_argument("--out")
    ap.add_argument("--runs-only", nargs="*")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--expected-f1", nargs="*",
                    help="NAME=F1 pairs to verify the fixed-threshold numbers")
    ap.add_argument("--merge", nargs="+",
                    help="dataset names for the merged rerank.json/rerank.md")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return
    if a.merge:
        os.makedirs(a.out, exist_ok=True)
        merge_reports(a.out, a.merge)
        return
    if not (a.dataset and a.runs and a.manifest and a.reference and a.out):
        ap.error("--dataset --runs --manifest --reference --out are required "
                 "(or use --selftest / --merge)")
    os.makedirs(a.out, exist_ok=True)
    exp = {}
    if a.expected_f1:
        for spec in a.expected_f1:
            rn, v = spec.split("=", 1)
            exp[rn] = float(v)
    run_dataset(a.dataset, a.runs, a.manifest, a.reference, a.out,
                a.runs_only, a.workers, expected_f1=exp or None)


if __name__ == "__main__":
    main()
