"""Detection metric for cerebral microbleeds after the official VALDO Task 2 rule.

Source: Sudre et al., "Where is VALDO? VAscular Lesions Detection and segmentatiOn
challenge at MICCAI 2021", arXiv 2208.07167, section 2.5 (Assessment method):
  - connected components with a neighbourhood of 6 for reference and prediction,
    prediction map thresholded at 0.5;
  - each reference element is matched to at most one predicted element; for
    microbleeds a match is possible when the centres of mass are less than 5 mm
    apart; among several candidates the one with the best association value
    (smallest centre distance) is attributed;
  - per case: F1 = 2TP/(2TP+FP+FN) and AED = |#ref - #pred| (absolute element
    difference); for cases without reference only the absolute metric is computed.
  - the paper reports median and interquartile range across cases.

Implementation choices (stated, not hidden):
  - Matching is ONE-TO-ONE and GREEDY BEST-FIRST: all (reference, prediction) pairs
    with centre distance < dist_mm are sorted by distance (ties: reference index,
    then prediction index) and accepted if neither side is taken yet. This is the
    closest-first reading of the paper, not a maximum-cardinality matching; in rare
    chains a globally optimal assignment could find one more TP.
  - Distances are in mm: voxel centroids times the linear part of the NIfTI affine
    (or diag(zooms) if only zooms are given). The array axes must be in the same
    order as the affine / zooms (as loaded by nibabel, no transposition).
  - Prediction binarised as pred > thr (strictly, same as cv5.filtert_m); for a
    binary {0,1} mask this equals >= 0.5. Reference binarised as ref > 0.5.
  - F1 is reported on 0..1 (paper: x100).
  - Pooled TP/FP/FN over ALL cases (including cases without reference, which
    contribute FP) are added next to the per-case median/IQR; they are not part of
    the paper's ranking.

Side information from the project metric (code/metric.py, imported unchanged):
  - m26_*: hits() under 26-connectivity and touch rule;
  - multi_hit: predicted 26-components that touch >= 2 reference 26-components
    (each counts as several TP and 0 FP under metric.hits -- the known loophole);
    refs_under_multi_hit: references touched by such components;
  - largest_pred_mm3: largest predicted 26-component in mm3 (0 if none).

Library:
    from metric_valdo import case_valdo, summarize_valdo, label_for
    row = case_valdo(ref_array, pred_array, zooms_or_affine, thr=0.5, dist_mm=5.0)
    summary = summarize_valdo([row, ...])
Usage:
    python metric_valdo.py --selftest
    python metric_valdo.py --pred <dir> [--labels '<path with {id}>'] [--thr 0.5]
                           [--dist 5] [--out summary.json] [--csv per_case.csv]
    --pred is searched recursively for *_pred.nii.gz (id = file name before _pred);
    without --labels the label is looked up in dev/gitter_c, dev/gitter_b,
    dev/preproc (first file on the SAME grid: shape and affine) -- mismatches abort.
"""
import os, sys, glob, json, argparse
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import hits, N26          # unchanged project metric (26-connectivity)

N6 = ndimage.generate_binary_structure(3, 1)
THR = 0.5
DIST_MM = 5.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_CANDIDATES = [f"{ROOT}/dev/gitter_c/{{id}}/{{id}}_label.nii.gz",
                    f"{ROOT}/dev/gitter_b/{{id}}/{{id}}_label.nii.gz",
                    f"{ROOT}/dev/preproc/{{id}}/{{id}}_label.nii.gz"]


def _lin(spacing):
    """3x3 matrix voxel index -> mm from zooms (3,) or an affine (4,4)/(3,3)."""
    a = np.asarray(spacing, dtype=float)
    if a.shape == (4, 4):
        return a[:3, :3]
    if a.shape == (3, 3):
        return a
    if a.shape == (3,):
        return np.diag(a)
    raise ValueError(f"spacing must be 3 zooms or a 4x4 affine, got shape {a.shape}")


def _centroids(lab, n):
    """Centre of mass (voxel index space) of labels 1..n -> (n,3) array, sizes (n,)."""
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0)
    nz = np.nonzero(lab)
    lv = lab[nz]
    cnt = np.bincount(lv, minlength=n + 1)[1:].astype(float)
    c = np.stack([np.bincount(lv, weights=ax.astype(float), minlength=n + 1)[1:] for ax in nz], 1)
    return c / cnt[:, None], cnt


def match_greedy(c_ref_mm, c_pred_mm, dist_mm=DIST_MM):
    """One-to-one greedy best-first matching by centre distance (< dist_mm, strict).
    Returns a list of (ref_index, pred_index, distance_mm), 0-based."""
    if len(c_ref_mm) == 0 or len(c_pred_mm) == 0:
        return []
    sp = cKDTree(c_ref_mm).sparse_distance_matrix(cKDTree(c_pred_mm), dist_mm, output_type="ndarray")
    if len(sp) == 0:
        return []
    r, p, d = sp["i"], sp["j"], sp["v"]
    keep = d < dist_mm
    r, p, d = r[keep], p[keep], d[keep]
    used_r, used_p, out = set(), set(), []
    for k in np.lexsort((p, r, d)):
        if r[k] in used_r or p[k] in used_p:
            continue
        used_r.add(int(r[k])); used_p.add(int(p[k]))
        out.append((int(r[k]), int(p[k]), float(d[k])))
    return out


def case_valdo(ref, pred, spacing, thr=THR, dist_mm=DIST_MM):
    """One case. ref/pred: arrays on the same grid; spacing: zooms (3,) or affine (4,4)."""
    ref_b = np.asarray(ref) > 0.5
    pred_b = np.asarray(pred) > thr
    if ref_b.shape != pred_b.shape:
        raise ValueError(f"grid mismatch: ref {ref_b.shape} vs pred {pred_b.shape}")
    M = _lin(spacing)
    vox_mm3 = float(abs(np.linalg.det(M)))
    # --- VALDO rule: 6-connectivity, centroid distance in mm, one-to-one ---
    lr, nr = ndimage.label(ref_b, structure=N6)
    lp, npd = ndimage.label(pred_b, structure=N6)
    cr, _ = _centroids(lr, nr)
    cp, _ = _centroids(lp, npd)
    m = match_greedy(cr @ M.T, cp @ M.T, dist_mm)
    tp = len(m); fp = int(npd) - tp; fn = int(nr) - tp
    f1 = round(2 * tp / (2 * tp + fp + fn), 4) if nr > 0 else None   # no reference -> absolute metric only
    # --- side information under the project metric (26-connectivity) ---
    h = hits(ref_b, pred_b)
    lr26, nr26 = ndimage.label(ref_b, structure=N26)
    lp26, np26 = ndimage.label(pred_b, structure=N26)
    assert np26 == h["n_pred"] and nr26 == h["n_ref"], "26-labelling disagrees with metric.hits"
    both = ref_b & pred_b
    multi, refs_multi = 0, 0
    if both.any():
        pairs = np.unique(np.stack([lp26[both], lr26[both]], 1), axis=0)
        per_pred = np.bincount(pairs[:, 0], minlength=np26 + 1)
        multi_ids = np.nonzero(per_pred >= 2)[0]
        multi = int(len(multi_ids))
        refs_multi = int(len(np.unique(pairs[np.isin(pairs[:, 0], multi_ids), 1])))
    largest = float(np.bincount(lp26.ravel())[1:].max() * vox_mm3) if np26 else 0.0
    return dict(n_ref=int(nr), n_pred=int(npd), tp=tp, fp=fp, fn=fn, f1=f1,
                aed=abs(int(nr) - int(npd)),
                match_dist_mm_max=round(max((x[2] for x in m), default=0.0), 3),
                multi_hit=multi, refs_under_multi_hit=refs_multi,
                largest_pred_mm3=round(largest, 3),
                m26_n_ref=h["n_ref"], m26_n_pred=h["n_pred"], m26_tp=h["tp"], m26_fp=h["fp"], m26_fn=h["fn"])


def _q(v, q):
    return round(float(np.percentile(v, q)), 4) if len(v) else None


def summarize_valdo(rows, thr=THR, dist_mm=DIST_MM):
    """Per-case F1 median + IQR (cases with reference only), AED over all cases, pooled
    TP/FP/FN over all cases, multi_hit and largest component, 26-rule sums alongside."""
    n = len(rows)
    f1s = [r["f1"] for r in rows if r["f1"] is not None]
    aeds = [r["aed"] for r in rows]
    s = {k: int(sum(r[k] for r in rows)) for k in ("n_ref", "n_pred", "tp", "fp", "fn")}
    p = s["tp"] / max(s["tp"] + s["fp"], 1); se = s["tp"] / max(s["tp"] + s["fn"], 1)
    m26 = {k: int(sum(r[f"m26_{k}"] for r in rows)) for k in ("n_ref", "n_pred", "tp", "fp", "fn")}
    return dict(
        rule=f"VALDO Task2 (arXiv 2208.07167 s2.5): 6-conn, pred>{thr}, centroid <{dist_mm} mm, "
             f"one-to-one greedy best-first",
        cases=n, cases_with_ref=len(f1s), cases_without_ref=n - len(f1s),
        f1_case_median=_q(f1s, 50), f1_case_q1=_q(f1s, 25), f1_case_q3=_q(f1s, 75),
        f1_case_mean=round(float(np.mean(f1s)), 4) if f1s else None,
        f1_case_std=round(float(np.std(f1s)), 4) if f1s else None,
        aed_median=_q(aeds, 50), aed_q1=_q(aeds, 25), aed_q3=_q(aeds, 75),
        aed_mean=round(float(np.mean(aeds)), 3) if aeds else None,
        aed_mean_cases_without_ref=round(float(np.mean([r["aed"] for r in rows if r["f1"] is None])), 3)
        if n - len(f1s) else None,
        **s,
        precision_pooled=round(p, 4), sensitivity_pooled=round(se, 4),
        f1_pooled=round(2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1), 4),
        multi_hit=int(sum(r["multi_hit"] for r in rows)),
        cases_with_multi_hit=int(sum(r["multi_hit"] > 0 for r in rows)),
        refs_under_multi_hit=int(sum(r["refs_under_multi_hit"] for r in rows)),
        largest_pred_mm3=max((r["largest_pred_mm3"] for r in rows), default=0.0),
        m26={**m26, "f1": round(2 * m26["tp"] / max(2 * m26["tp"] + m26["fp"] + m26["fn"], 1), 4)},
    )


def label_for(sid, pred_img, template=None, candidates=None):
    """Label NIfTI on the SAME grid (shape + affine) as pred_img. Raises if none fits.
    template: one path with {id}; candidates: list of such paths (first on-grid wins)."""
    import nibabel as nib
    tried = []
    for t in ([template] if template else (candidates or LABEL_CANDIDATES)):
        f = t.format(id=sid)
        if not os.path.exists(f):
            tried.append(f"{f} (missing)"); continue
        li = nib.load(f)
        if li.shape[:3] == pred_img.shape[:3] and np.allclose(li.affine, pred_img.affine, atol=1e-3):
            return li
        tried.append(f"{f} (other grid {li.shape[:3]})")
    raise SystemExit(f"{sid}: no label on the prediction grid {pred_img.shape[:3]}; tried: {tried}")


def evaluate_dir(pred_dir, labels=None, thr=THR, dist_mm=DIST_MM, out=None, csv_path=None):
    import nibabel as nib, csv
    files = sorted(glob.glob(os.path.join(pred_dir, "**", "*_pred.nii.gz"), recursive=True))
    if not files:
        raise SystemExit(f"no *_pred.nii.gz under {pred_dir}")
    ids = [os.path.basename(f)[: -len("_pred.nii.gz")] for f in files]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise SystemExit(f"cases predicted more than once: {dup[:10]}")
    rows = []
    for sid, f in zip(ids, files):
        pi = nib.load(f)
        li = label_for(sid, pi, labels)
        r = case_valdo(np.asanyarray(li.dataobj), np.asanyarray(pi.dataobj), pi.affine, thr, dist_mm)
        rows.append(dict(id=sid, **r))
    z = summarize_valdo(rows, thr, dist_mm)
    z["pred_dir"] = os.path.abspath(pred_dir)
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if out:
        tmp = out + ".tmp"
        json.dump(z, open(tmp, "w"), indent=1); os.replace(tmp, out)
    print(json.dumps(z))
    return z, rows


def selftest():
    def ball(a, c, r, v=1.0):
        z, y, x = np.ogrid[:a.shape[0], :a.shape[1], :a.shape[2]]
        a[((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) <= r * r] = v
    iso = (1.0, 1.0, 1.0)

    # 1) blob over two references: VALDO at most 1 TP; metric.hits 2 TP 0 FP; multi_hit 1
    ref = np.zeros((40, 40, 40), np.uint8); pred = np.zeros_like(ref)
    ball(ref, (20, 20, 17), 1); ball(ref, (20, 20, 23), 1)       # centres 6 mm apart, both 3 mm from blob centre
    ball(pred, (20, 20, 20), 5)
    r = case_valdo(ref, pred, iso)
    assert (r["n_ref"], r["n_pred"], r["tp"], r["fp"], r["fn"]) == (2, 1, 1, 0, 1), r
    assert r["m26_tp"] == 2 and r["m26_fp"] == 0 and r["multi_hit"] == 1 and r["refs_under_multi_hit"] == 2, r
    assert abs(r["f1"] - 2 / 3) < 1e-3 and r["aed"] == 1, r
    assert abs(r["largest_pred_mm3"] - float((pred > 0).sum())) < 1e-6, r

    # 2) distance in mm, anisotropic zooms (1,1,4): 2 voxels along axis 2 = 8 mm (no match),
    #    2 voxels along axis 0 = 2 mm (match)
    ref = np.zeros((30, 30, 12), np.uint8); pred = np.zeros_like(ref)
    ref[10, 10, 3] = 1; pred[10, 10, 5] = 1          # 8 mm -> FN + FP
    ref[20, 20, 8] = 1; pred[22, 20, 8] = 1          # 2 mm -> TP
    r = case_valdo(ref, pred, (1.0, 1.0, 4.0))
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1), r
    aff = np.diag([1.0, 1.0, 4.0, 1.0]); aff[:3, 3] = (-7, 3, 11)   # affine gives the same distances
    assert case_valdo(ref, pred, aff) == r
    assert case_valdo(ref, pred, (1.0, 1.0, 1.0))["tp"] == 2      # isotropic: both within 5 mm

    # 3) 6- vs 26-connectivity: two voxels touching only at a corner = 2 elements under VALDO
    ref = np.zeros((10, 10, 10), np.uint8); pred = np.zeros_like(ref)
    ref[4, 4, 4] = 1; pred[4, 4, 4] = 1; pred[5, 5, 5] = 1
    r = case_valdo(ref, pred, iso)
    assert (r["n_pred"], r["tp"], r["fp"], r["m26_n_pred"], r["m26_fp"]) == (2, 1, 1, 1, 0), r

    # 4) threshold: probability 0.4 ignored, 0.6 counted; exactly 0.5 ignored (strict >)
    ref = np.zeros((20, 20, 20), np.uint8); prob = np.zeros((20, 20, 20), np.float32)
    ref[5, 5, 5] = 1; prob[5, 5, 5] = 0.6; prob[15, 15, 15] = 0.4; prob[10, 2, 2] = 0.5
    r = case_valdo(ref, prob, iso)
    assert (r["n_pred"], r["tp"], r["fp"]) == (1, 1, 0), r

    # 5) one-to-one greedy best-first: pred A 1 mm from ref1 and 3 mm from ref2, pred B 4 mm
    #    from ref2 -> (A,ref1) then (B,ref2): 2 TP; a second pred on ref1 stays FP
    ref = np.zeros((30, 30, 30), np.uint8); pred = np.zeros_like(ref)
    ref[10, 10, 10] = 1; ref[10, 10, 14] = 1
    pred[10, 10, 11] = 1; pred[10, 10, 18] = 1; pred[10, 13, 10] = 1
    r = case_valdo(ref, pred, iso)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 0), r

    # 6) empty reference: F1 not applicable, AED only; empty/empty: AED 0
    e1 = case_valdo(np.zeros((8, 8, 8)), np.pad(np.ones((1, 1, 1)), ((3, 4), (3, 4), (3, 4))), iso)
    e0 = case_valdo(np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), iso)
    assert e1["f1"] is None and e1["aed"] == 1 and e1["fp"] == 1 and e0["f1"] is None and e0["aed"] == 0, (e1, e0)

    # 7) summary: median/IQR only over cases with reference, pooled sums over all
    rows = [case_valdo(*a) for a in [
        (np.pad(np.ones((1, 1, 1)), 4), np.pad(np.ones((1, 1, 1)), 4), iso),   # F1 1
        (np.pad(np.ones((1, 1, 1)), 4), np.zeros((9, 9, 9)), iso),              # F1 0
    ]] + [e1, e0]
    z = summarize_valdo(rows)
    assert z["cases"] == 4 and z["cases_with_ref"] == 2 and z["f1_case_median"] == 0.5, z
    assert (z["tp"], z["fp"], z["fn"]) == (1, 1, 1) and z["aed_mean_cases_without_ref"] == 0.5, z
    print("selftest passed: blob-on-2-refs VALDO", (1, 0, 1), "vs metric.hits 2 TP/0 FP, multi_hit 1;",
          "mm distances, 6-conn, threshold, greedy one-to-one, empty cases, summary ok |", json.dumps(z))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pred"); ap.add_argument("--labels")
    ap.add_argument("--thr", type=float, default=THR); ap.add_argument("--dist", type=float, default=DIST_MM)
    ap.add_argument("--out"); ap.add_argument("--csv")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.pred:
        evaluate_dir(a.pred, a.labels, a.thr, a.dist, a.out, a.csv)
    else:
        ap.print_help()
