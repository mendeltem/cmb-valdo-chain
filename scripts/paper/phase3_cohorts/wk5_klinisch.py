"""wk5 (31.11.2026): MARS location + patient-level clinical endpoints, all CPU.

Inputs
  * candidate folders (reference counts): valdo ergebnisse/stage2/a03-aniso-e60-gd-weich-z40,
    t2star <private>/stage2/a03-aniso-e60-cat-t2s-weich-z40, swi <private>/stage2/a03-aniso-e60-cat-swi-v2-weich-z40
  * reference labels: <dir>/<case>_label.nii.gz (26-connected components = the reference CMB, index 1..n)
  * SynthSeg: valdo dev/synthseg/{sid}_synthseg.nii.gz, t2star <private>/synthseg-t2star/robust,
    swi <private>/synthseg-swi/robust
  * decision dumps: <private>/paper/decisions/chainA_s{0,1}.jsonl (mode 'pooled tset-plain' = chain A,
    'probability+csf' = chain B), chainC_s{0,1}.jsonl (mode 'pooled tset-ctx' = chain C, queued 01.11.)

Outputs (aggregates only, no per-case values)
  MB/paper/mars_summary.json, MB/paper/clinical_summary.json, MB/paper/wk5_ref_cache.npz,
  MB/paper/mars_qc/index.html + pngs, P/fig_bland_altman.png, P/fig_therapy_threshold.png,
  P/fig_caa_pattern.png.

MARS classes: cmb.analysis.mars_location (pre-specified, Gregoire 2009). Endpoints:
a) count agreement (Spearman, ICC(3,1), Bland-Altman), b) presence, c) burden 0/1-4/>=5,
d) therapy '> 4', e) Boston pattern (CMB-only proxy), f) per MARS class recall/precision + size.
Case-bootstrap 2000 draws, stratified by dataset.
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

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import nibabel as nib
import nibabel.processing
from scipy import ndimage

V = _R
MB = _P
P = f"{V}/paper"
sys.path.insert(0, V)
from cmb.analysis.mars_location import (CEREBELLUM, BRAINSTEM, DEEP_GRAY, WM, CORTEX, CSF,
                                        OTHER, classify_lesion)                       # noqa: E402
import importlib.util
_spec = importlib.util.spec_from_file_location("gewebekarte", f"{V}/code/gewebekarte.py")
_gw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_gw)
ort_der_komponente, ALTE_ORT = _gw.ort_der_komponente, _gw.ORT

CAND = {
    "valdo": f"{V}/ergebnisse/stage2/a03-aniso-e60-gd-weich-z40/candidates.json",
    "t2star": f"{MB}/stage2/a03-aniso-e60-cat-t2s-weich-z40/candidates.json",
    "swi": f"{MB}/stage2/a03-aniso-e60-cat-swi-v2-weich-z40/candidates.json",
}
GRID_DIR = {
    "valdo": lambda cid: f"{V}/dev/gitter_d/{cid}",
    "t2star": lambda cid: f"{MB}/gitter/t2star/{cid}",
    "swi": lambda cid: f"{MB}/gitter/swi_v2/{cid}",
}
SEG_PATH = {
    "valdo": lambda cid: f"{V}/dev/synthseg/{cid}_synthseg.nii.gz",
    "t2star": lambda cid: f"{MB}/synthseg-t2star/robust/{cid.split('-', 1)[1]}_synthseg.nii.gz",
    "swi": lambda cid: f"{MB}/synthseg-swi/robust/{cid.split('-', 1)[1]}_synthseg.nii.gz",
}
COHORTS = ("valdo", "t2star", "swi")
N_BOOT = 2000
BOSTON_ORDER = ("none_or_single", "deep_only", "mixed", "strictly_lobar")
BOSTON_W = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], float)
BOSTON_W = BOSTON_W / BOSTON_W.max()
SIZE_ORDER = ("1_<2mm", "2_2-3mm", "3_3-5mm", "4_5-10mm")
BURDEN = (0, 1, 2)
BURDEN_W = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], float)

# ------------------------------------------------------------------------------ stats
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    def rank(z):
        o = np.argsort(z, kind="mergesort")
        r = np.empty(len(z), float); r[o] = np.arange(1, len(z) + 1)
        i = 0
        while i < len(z):
            j = i
            while j + 1 < len(z) and z[o[j + 1]] == z[o[i]]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def icc31(x, y):
    """ICC(3,1): two raters (reference, model), one-way random effects, absolute agreement."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan")
    g = np.stack([x, y], axis=1)
    ss_b = n * np.sum((g.mean(axis=1) - g.mean()) ** 2)
    ss_w = np.sum((g - g.mean(axis=1, keepdims=True)) ** 2)
    msb, msw = ss_b / 1, ss_w / (n - 1)
    if np.isclose(msb, msw):
        return 0.0 if msw == 0 else float("nan")
    return float((msb - msw) / (msb + (n - 1) * msw))


def cohens_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    if len(cats) < 2:
        return float("nan")
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    if pe >= 1:
        return float("nan")
    return float((po - pe) / (1 - pe))


def weighted_kappa(a, b, w):
    a, b = np.asarray(a, int), np.asarray(b, int)
    obs = np.zeros(w.shape, float)
    for i, j in zip(a, b):
        obs[i, j] += 1
    n = obs.sum()
    ra = obs.sum(1) / n; cb = obs.sum(0) / n
    denom = (w * np.outer(ra, cb)).sum()
    if n == 0 or denom == 0:
        return float("nan")
    return float(1 - (w * (obs / n)).sum() / denom)


def confusion(ref, pred, cats):
    cm = np.zeros((len(cats), len(cats)), int)
    for i, j in zip(ref, pred):
        if i in cats and j in cats:
            cm[cats.index(i), cats.index(j)] += 1
    return cm


def boot_pairs(rows, stat, n_boot=N_BOOT, seed=20261101):
    """rows: list of per-case dicts with keys 'cohort', 'ref', 'pred'. Resample cases within cohort,
    recompute stat on the pooled resample. -> (lo, hi)."""
    rng = np.random.RandomState(seed)
    by = {c: [r for r in rows if r["cohort"] == c] for c in COHORTS}
    vals = []
    for _ in range(n_boot):
        picks = [[by[c][i] for i in rng.randint(0, len(by[c]), len(by[c]))] for c in COHORTS if by[c]]
        a = np.concatenate([np.array([p["ref"] for p in part], float) for part in picks])
        b = np.concatenate([np.array([p["pred"] for p in part], float) for part in picks])
        vals.append(stat(a, b))
    vals = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    return tuple(float(q) for q in np.percentile(vals, [2.5, 97.5]))


def rnd(x, d=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def ci(x):
    return None if x is None or not np.isfinite(x[0]) else [rnd(x[0]), rnd(x[1])]


def burden(n):
    return 0 if n == 0 else 1 if n <= 4 else 2


def boston_pattern(n_lobar, n_deep, n_brainstem, n_cereb, cerebellum_as_deep=False):
    """Pre-specified (Boston v2.0 imaging part, CMB-only proxy).
    Main: cerebellar CMBs are excluded from the pattern. Sensitivity: counted as deep.
    'none/single' = no CMB or a single CMB that cannot define a pattern (1 lobar or 1 deep/brainstem)."""
    lob, deep, brain = int(n_lobar), int(n_deep), int(n_brainstem)
    if cerebellum_as_deep:
        deep += int(n_cereb)
    if lob + deep + brain == 0:
        return "none_or_single"
    if lob + deep + brain == 1:
        return "none_or_single"
    if deep == 0 and brain == 0 and lob >= 2:
        return "strictly_lobar"
    if lob >= 1 and deep >= 1:
        return "mixed"
    if lob == 0 and (deep + brain) >= 2:
        return "deep_only"
    if lob == 1 and brain >= 1:
        # one lobar + brainstem, no deep: not 'strictly lobar', not 'mixed' (needs deep) -> deep_only side
        return "deep_only"
    if lob >= 2 and brain >= 1:
        return "mixed"
    if lob >= 1:
        return "none_or_single"
    return "deep_only"


# ------------------------------------------------------------------------------ data
_SEG_CACHE: dict = {}


def load_synthseg_like(path, like):
    key = path
    if key in _SEG_CACHE:
        return _SEG_CACHE[key]
    seg = nib.load(path)
    if seg.shape != like.shape:
        seg = nibabel.processing.resample_from_to(seg, like, order=0)
    arr = np.asarray(seg.dataobj).astype(np.int32)
    _SEG_CACHE[key] = arr
    return arr


def load_reference(cohort, cid):
    """-> (n_ref, per-lesion list: mars, sub, band, size, alte, majority)."""
    d = GRID_DIR[cohort](cid)
    img = nib.load(f"{d}/{cid}_image.nii.gz")
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    ref = np.asarray(img.__class__(np.asarray(nib.load(f"{d}/{cid}_label.nii.gz").dataobj), img.affine).dataobj) > 0
    labels, n = ndimage.label(ref, structure=np.ones((3, 3, 3), bool))
    out = []
    if n:
        p = SEG_PATH[cohort](cid)
        seg = load_synthseg_like(p, img) if os.path.exists(p) else None
        boxes = ndimage.find_objects(labels)
        for i in range(1, n + 1):
            box = tuple(slice(max(b.start - 4, 0), min(b.stop + 4, labels.shape[a]))
                        for a, b in enumerate(boxes[i - 1]))
            mask = labels[box] == i
            if seg is not None:
                r = classify_lesion(seg[box], mask, spacing)
                mars, sub, band = r["mars"], r["sub"], bool(r["band5mm_lobar"])
            else:
                mars, sub, band = "other", "no_synthseg", False
            zz = np.nonzero(mask)
            xy = (zz[0].max() - zz[0].min() + 1) * spacing[0]
            yz = (zz[1].max() - zz[1].min() + 1) * spacing[1]
            diam = max(xy, yz)
            size = "1_<2mm" if diam < 2 else "2_2-3mm" if diam < 3 else "3_3-5mm" if diam < 5 else "4_5-10mm"
            alte = ort_der_komponente(mask, seg[box]) if seg is not None else "ausserhalb"
            out.append(dict(mars=mars, sub=sub, band=band, size=size, alte=alte))
    return n, out


# ------------------------------------------------------------------------------ figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 9,
                     "axes.edgecolor": "black"})


def fig_bland_altman(points, path):
    ref = np.array([p[0] for p in points], float)
    pred = np.array([p[1] for p in points], float)
    d = pred - ref
    med = float(np.median(d))
    lo, hi = np.percentile(d, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(ref, d, s=14, color="#1f4e79", alpha=0.8, edgecolors="none", zorder=3)
    for y, c, lab, va in [(0, "k", "y = 0", "bottom"), (med, "#b22222", f"median difference = {med:.2f}", "top"),
                          (lo, "gray", "2.5 %", "bottom"), (hi, "gray", "97.5 %", "top")]:
        ax.axhline(y, color=c, lw=0.9, ls="-" if y == med else "--", zorder=1)
        ax.text(np.percentile(ref, 96) * 0.99, y + (0.06 * np.ptp(d) if va == "bottom" else -0.06 * max(np.ptp(d), 1)),
                lab, fontsize=8, va=va, ha="right")
    ax.set_xlabel("reference CMB count (per patient)")
    ax.set_ylabel("model $-$ reference (count)")
    ax.set_title("Bland-Altman: chain A (seeds 0 and 1, averaged per case) vs reference", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _confusion_axes(ax, labels, cm, title):
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("model"); ax.set_ylabel("reference")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_title(title, fontsize=9)


def fig_therapy(path, data):
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
    labels = ["<= 4", "> 4"]          # fix 12.11.2026: row/col 0 is <= 4 (tn, fp), the labels were swapped
    for ax, coh in zip(axes, list(data.keys())[:4]):
        d = data[coh]
        cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])     # rows reference, cols model
        _confusion_axes(ax, labels, cm, coh)
    fig.suptitle("Anti-amyloid therapy threshold: exclusion if count > 4 (rows = reference, cols = chain A)",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_caa(path, data):
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
    for ax, coh in zip(axes, list(data.keys())[:4]):
        _confusion_axes(ax, BOSTON_ORDER, np.asarray(data[coh], int), coh)
    fig.suptitle("CAA pattern, CMB-only proxy of the Boston v2.0 imaging part (chain A; rows = reference)",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="only this many cases per cohort (timing check)")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--skip-mars-qc", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print(f"start {time.strftime('%H:%M:%S')} smoke={args.smoke}", flush=True)

    # ---- reference side (cached)
    cache_p = f"{MB}/paper/wk5_ref_cache.npz"
    cand_meta = {c: json.load(open(CAND[c]))["cases"] for c in COHORTS}
    if os.path.exists(cache_p) and not args.smoke:
        z = np.load(cache_p, allow_pickle=True)
        ref = {c: {str(k): (int(z[f"n_{c}"][i]), list(z[f"les_{c}"][i]))
                   for i, k in enumerate(z[f"ids_{c}"])} for c in COHORTS}
        print(f"reference cache loaded in {time.time() - t0:.0f} s", flush=True)
    else:
        ref = {c: {} for c in COHORTS}
        for c in COHORTS:
            items = list(cand_meta[c].items())
            if args.smoke:
                items = items[:args.smoke]
            for i, (cid, meta) in enumerate(items):
                n, les = load_reference(c, cid)
                assert n == meta["n_reference"], (c, cid, n, meta["n_reference"])
                ref[c][cid] = (n, les)
                if (i + 1) % 5 == 0 or i == len(items) - 1:
                    print(f"  {c}: {i + 1}/{len(items)} cases, {time.time() - t0:.0f} s", flush=True)
        if not args.smoke:
            np.savez(cache_p, **{f"ids_{c}": np.array(list(ref[c].keys())) for c in COHORTS},
                     **{f"n_{c}": np.array([ref[c][k][0] for k in ref[c]]) for c in COHORTS},
                     **{f"les_{c}": np.array([ref[c][k][1] for k in ref[c]], dtype=object) for c in COHORTS})
            print(f"reference cache written in {time.time() - t0:.0f} s", flush=True)

    # ---- decision dumps
    dumps = {}
    for s in (0, 1):
        p = f"{MB}/paper/decisions/chainA_s{s}.jsonl"
        if os.path.exists(p):
            dumps[("A", s)] = [json.loads(l) for l in open(p)]
    for s in (0, 1):
        p = f"{MB}/paper/decisions/chainC_s{s}.jsonl"
        if os.path.exists(p):
            dumps[("C", s)] = [json.loads(l) for l in open(p)]
    chains = {}
    if ("A", 0) in dumps:
        chains["A"] = ("pooled tset-plain", [s for s in (0, 1) if ("A", s) in dumps])
    if ("A", 0) in dumps:
        chains["B"] = ("probability+csf", [s for s in (0, 1) if ("A", s) in dumps])
    if ("C", 0) in dumps:
        chains["C"] = ("pooled tset-ctx", [s for s in (0, 1) if ("C", s) in dumps])
    print("chains available: " + ", ".join(f"{k} (mode {v[0]}, seeds {v[1]})" for k, v in chains.items()), flush=True)

    # accepted candidates per (chain, seed, cohort, case) -> list of dump rows
    acc = {k: {c: {} for c in COHORTS} for k in chains}
    for key, (mode, seeds) in chains.items():
        src = "A" if key in ("A", "B") else key
        for s in seeds:
            for r in dumps[(src, s)]:
                if r["mode"] == mode and r["accepted"]:
                    acc[key][r["cohort"]].setdefault(r["case"], []).append(r)

    # model count per (chain, cohort, case) = mean over seeds
    def model_counts(key):
        out = {}
        for c in COHORTS:
            allc = set(ref[c])
            for cid in allc:
                # 01.11.2026: acc holds the accepted rows of ALL seeds in one list; len() was the SUM over the
                # seeds (counts doubled with two seeds). Count per seed via seed_offset, then average.
                ns = [sum(1 for r in acc[key][c].get(cid, []) if r.get("seed_offset", s) == s) for s in chains[key][1]]
                out[cid] = float(np.mean(ns))
        return out

    # ---- MARS summary: reference side
    n_ref_total = 0
    ref_dist = {}
    old_counts, moved_ld, moved_dl, band_n, cereb_ref, brain_ref = {}, 0, 0, 0, 0, 0
    ref_case_mars = {}
    for c in COHORTS:
        for cid, (n, les) in ref[c].items():
            n_ref_total += n
            mc = {k: 0 for k in ("lobar", "deep", "brainstem", "cerebellar", "other")}
            for l in les:
                key2 = l["mars"] + (f" ({l['sub']})" if l["sub"] in ("brainstem", "cerebellum") else "")
                ref_dist[key2] = ref_dist.get(key2, 0) + 1
                old_counts[l["alte"]] = old_counts.get(l["alte"], 0) + 1
                if l["mars"] == "lobar" and l["alte"] == "tief":
                    moved_ld += 1
                if l["mars"] == "deep" and l["alte"] == "lobaer":
                    moved_dl += 1
                if l["band"]:
                    band_n += 1
                if l["sub"] == "cerebellum":
                    cereb_ref += 1; mc["cerebellar"] += 1
                if l["sub"] == "brainstem":
                    brain_ref += 1; mc["brainstem"] += 1
                if l["mars"] in mc and l["sub"] not in ("brainstem", "cerebellum"):
                    mc[l["mars"]] += 1
            ref_case_mars[(c, cid)] = mc
    # MARS 4-class mapping for the pattern: infratentorial -> brainstem / cerebellar (sub-class)
    def _pat4(mc):
        return (mc["lobar"], mc["deep"], mc["brainstem"], mc["cerebellar"])
    print(f"reference: {n_ref_total} CMB, {time.time() - t0:.0f} s", flush=True)

    # ---- chain A per MARS class / size (prediction side, seeds 0+1 averaged where two exist)
    cand_lists = {c: json.load(open(CAND[c]))["candidates"] for c in COHORTS}
    per_class = {c: {"n_ref": 0, "found": 0, "pred": 0, "pred_true": 0} for c in
                 ("lobar", "deep", "infratentorial", "other")}
    per_size = {s: {"n_ref": 0, "found": 0} for s in SIZE_ORDER}
    for c in COHORTS:
        for cid, (n, les) in ref[c].items():
            touched_union = set()
            for s in chains.get("A", (None, []))[1] or []:
                for r in acc["A"][c].get(cid, []):
                    for t in r["touched"]:
                        touched_union.add(t)
            for j, l in enumerate(les, 1):
                per_class[l["mars"]]["n_ref"] += 1
                per_size[l["size"]]["n_ref"] += 1
                if j in touched_union:
                    per_class[l["mars"]]["found"] += 1
                    per_size[l["size"]]["found"] += 1
    # predicted side: classify each accepted candidate (mean over seeds: a candidate counts once if accepted in any seed)
    maj_infra = set(CEREBELLUM) | set(BRAINSTEM)
    maj_deep = set(DEEP_GRAY)
    maj_cort = set(CORTEX)
    for c in COHORTS:
        seen = set()
        for s in chains.get("A", (None, []))[1] or []:
            for cid in ref[c]:
                for r in acc["A"][c].get(cid, []):
                    k = (cid, tuple(r["centre_voxel"]), r["voxels"])
                    if k in seen:
                        continue
                    seen.add(k)
                    t = r["tissue"]
                    if t in maj_infra:
                        mc = "infratentorial"
                    elif t in maj_deep:
                        mc = "deep"
                    elif t in maj_cort or t in set(WM):
                        mc = _wm_class(c, r)
                    else:
                        mc = "other"
                    per_class[mc]["pred"] += 1
                    if r["touched"]:
                        per_class[mc]["pred_true"] += 1

    mars_summary = {
        "date": datetime.date.today().isoformat(),
        "definition": "MARS (Gregoire et al. 2009, Neurology 73:1759-66, doi:10.1212/WNL.0b013e3181c34a7d), "
                      "operationalised pre-specified on the SynthSeg labels already used by cmb.stage2.candidates; "
                      "not tuned. Lobe not reported: no SynthSeg --parc maps exist in this project.",
        "n_reference_cmb": int(n_ref_total),
        "reference_distribution": dict(sorted(ref_dist.items())),
        "band5mm_lobar_sensitivity": {"definition": "fixed 5 mm juxtacortical band instead of the distance rule",
                                       "n_lobar_by_band": int(band_n)},
        "agreement_with_old_synthseg_majority": {
            "old_counts": old_counts,
            "moved_lobar_to_deep": int(moved_ld),
            "moved_deep_to_lobar": int(moved_dl),
        },
        "chainA_per_mars_class": {},
        "chainA_per_size_class": {},
        "chainA_seeds": chains.get("A", (None, []))[1],
    }
    for c in per_class:
        d = per_class[c]
        mars_summary["chainA_per_mars_class"][c] = {
            "n_reference": int(d["n_ref"]), "found": int(d["found"]),
            "n_predicted": int(d["pred"]), "predicted_on_reference": int(d["pred_true"]),
            "recall": rnd(d["found"] / d["n_ref"]) if d["n_ref"] else None,
            "precision": rnd(d["pred_true"] / d["pred"]) if d["pred"] else None,
        }
    for s in SIZE_ORDER:
        d = per_size[s]
        mars_summary["chainA_per_size_class"][s] = {
            "n_reference": int(d["n_ref"]), "found": int(d["found"]),
            "recall": rnd(d["found"] / d["n_ref"]) if d["n_ref"] else None,
        }

    # ---- patient-level endpoints
    clinical = {
        "date": datetime.date.today().isoformat(),
        "note": "Aggregates only. Case-bootstrap 2000 draws stratified by dataset. "
                "Own-cohort data: no case identifiers, no per-case values. "
                "Chain A = specialised U-Nets + pooled tset-plain (SWI v2 candidates, seed offsets 0 and 1, "
                "averaged per case where two seeds exist, both reported in seeds). "
                "Chain B = stage 1 alone, probability + CSF rule (same dumps). "
                "Chain C = merged sensitive stage 1 + soft context (tset-ctx on the union candidates).",
        "human_benchmark": "MARS inter-rater ICC 0.93 and kappa 0.68 for presence (Gregoire et al. 2009)",
        "chains": {},
    }
    for key, (mode, seeds) in chains.items():
        mc_counts = model_counts(key)
        entry = {"seeds": seeds, "by_dataset": {}, "all": {}}
        for scope, cohorts in (("by_dataset", {c: [c] for c in COHORTS}),):
            for c, _ in cohorts.items():
                rows = [{"cohort": c, "ref": float(ref[c][cid][0]), "pred": mc_counts[cid]}
                        for cid in ref[c]]
                entry[scope][c] = _endpoints(rows)
        rows = [{"cohort": c, "ref": float(ref[c][cid][0]), "pred": mc_counts[cid]}
                for c in COHORTS for cid in ref[c]]
        entry["all"] = _endpoints(rows)
        clinical["chains"][key] = entry

    # therapy threshold + CAA pattern for chain A (per-case, then aggregated per scope)
    caa_main = caa_sens = None
    therapy = None
    if "A" in chains:
        mc_counts = model_counts("A")
        ref4, pred4, refp_all, predp_all, predps_all = [], [], [], [], []
        for c in COHORTS:
            for cid in ref[c]:
                rn, mn = ref[c][cid][0], mc_counts[cid]
                ref4.append(rn > 4); pred4.append(mn > 4)
                mm = ref_case_mars[(c, cid)]
                refp_all.append(boston_pattern(*_pat4(mm)))
                # model pattern: MARS classes of its accepted candidates (seeds union, dedup)
                seen, mcls = set(), {"lobar": 0, "deep": 0, "brainstem": 0, "cerebellar": 0, "other": 0}
                for s in chains["A"][1]:
                    for r in acc["A"][c].get(cid, []):
                        k = (tuple(r["centre_voxel"]), r["voxels"])
                        if k in seen:
                            continue
                        seen.add(k)
                        mcls[_model_mars(c, r)] += 1
                predp_all.append(boston_pattern(mcls["lobar"], mcls["deep"], mcls["brainstem"], 0, False))
                predps_all.append(boston_pattern(mcls["lobar"], mcls["deep"], mcls["brainstem"], mcls["cerebellar"], True))

        def _therapy_block(sel):
            a4, p4 = np.asarray(sel[0]), np.asarray(sel[1])
            tp = int(np.sum(a4 & p4)); fp = int(np.sum(~a4 & p4)); fn = int(np.sum(a4 & ~p4)); tn = int(np.sum(~a4 & ~p4))
            return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "sensitivity": rnd(tp / max(tp + fn, 1)), "specificity": rnd(tn / max(tn + fp, 1)),
                    "wrongly_eligible": fn, "wrongly_excluded": fp}

        def _scope_indices(scope):
            idx = []
            i = 0
            for c in COHORTS:
                for cid in ref[c]:
                    if scope in ("all", c):
                        idx.append(i)
                    i += 1
            return idx
        therapy, caa_main, caa_sens = {}, {}, {}
        for scope in COHORTS + ("all",):
            idx = _scope_indices(scope)
            therapy[scope] = _therapy_block((np.array([ref4[i] for i in idx]),
                                             np.array([pred4[i] for i in idx])))
            caa_main[scope] = confusion([refp_all[i] for i in idx], [predp_all[i] for i in idx], BOSTON_ORDER).tolist()
            caa_sens[scope] = confusion([refp_all[i] for i in idx], [predps_all[i] for i in idx], BOSTON_ORDER).tolist()

        def _wk_of(cm_lists):
            return {k: rnd(weighted_kappa(np.array([BOSTON_ORDER.index(p) for p in rr]),
                                          np.array([BOSTON_ORDER.index(p) for p in pp]), BOSTON_W))
                    for k, rr, pp in cm_lists}

        clinical["therapy_threshold"] = {
            "rule": "exclusion if > 4 CMB (donanemab appropriate use recommendations, Rabinovici et al. 2025, "
                    "J Prev Alzheimers Dis 12(5):100150, doi:10.1016/j.tjpad.2025.100150); chain A",
            "chains_A": therapy,
        }
        clinical["caa_pattern"] = {
            "rule": "imaging part only of the Boston criteria v2.0 (Charidimou et al. 2022, Lancet Neurol 21:714-25, "
                    "doi:10.1016/S1474-4422(22)00208-3). CMB-only proxy, not a CAA diagnosis: no ICH, siderosis, "
                    "white matter features or age are available. 'strictly lobar' = >= 2 lobar and no deep/brainstem; "
                    "'mixed' = lobar and deep; 'deep only'; 'none/single' = no CMB or a single CMB. "
                    "Cerebellar CMBs: pre-specified as EXCLUDED from the pattern, sensitivity analysis counting them "
                    "as deep -- this choice is to be confirmed by a clinical co-author.",
            "order": list(BOSTON_ORDER),
            "chainA_confusion_reference_vs_model": caa_main,
            "weighted_kappa": _wk_of([(s, [refp_all[i] for i in _scope_indices(s)],
                                       [predp_all[i] for i in _scope_indices(s)]) for s in COHORTS + ("all",)]),
            "cerebellar_as_deep_sensitivity": {
                "confusion": caa_sens,
                "weighted_kappa": _wk_of([(s, [refp_all[i] for i in _scope_indices(s)],
                                           [predps_all[i] for i in _scope_indices(s)]) for s in COHORTS + ("all",)])},
            "reference_pattern_distribution": _dist_of(refp_all),
            "model_pattern_distribution": _dist_of(predp_all),
        }

    if not args.no_figs and "A" in chains:
        mc_counts = model_counts("A")
        pts = [(ref[c][cid][0], mc_counts[cid]) for c in COHORTS for cid in ref[c]]
        # 12.11.2026: per-patient count pairs for Fig. 4 (cohort, reference, chain mean of the seeds); no case ids
        json.dump({"what": "per-patient CMB counts, chain A (pooled tset-plain = frozen-V), seeds averaged; no ids",
                   "pairs": [[c, int(ref[c][cid][0]), float(mc_counts[cid])] for c in COHORTS for cid in sorted(ref[c])]},
                  open(f"{MB}/paper/wk5_count_pairs.json", "w"))
        fig_bland_altman(pts, f"{P}/fig_bland_altman.png")
        fig_therapy(f"{P}/fig_therapy_threshold.png", therapy)
        fig_caa(f"{P}/fig_caa_pattern.png", caa_main)
        print(f"figures written in {time.time() - t0:.0f} s", flush=True)

    if not args.no_figs:
        json.dump(mars_summary, open(f"{MB}/paper/mars_summary.json", "w"), indent=1)
        json.dump(clinical, open(f"{MB}/paper/clinical_summary.json", "w"), indent=1)
        print(f"summaries written in {time.time() - t0:.0f} s", flush=True)

    print(f"DONE wk5_klinisch in {time.time() - t0:.0f} s", flush=True)


def _endpoints(rows):
    ref = np.array([r["ref"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    out = {"n": len(rows)}
    out["count"] = {
        "spearman_rho": rnd(spearman(ref, pred)),
        "spearman_ci": ci(boot_pairs(rows, spearman)),
        "icc_31": rnd(icc31(ref, pred)),
        "icc_ci": ci(boot_pairs(rows, icc31)),
        "bland_altman": {"median_diff": rnd(float(np.median(pred - ref))),
                          "limit_2.5pct": rnd(float(np.percentile(pred - ref, 2.5))),
                          "limit_97.5pct": rnd(float(np.percentile(pred - ref, 97.5)))},
    }
    r1, p1 = ref >= 1, pred >= 1
    tp = int(np.sum(r1 & p1)); fp = int(np.sum(~r1 & p1)); fn = int(np.sum(r1 & ~p1)); tn = int(np.sum(~r1 & ~p1))
    out["presence"] = {"sensitivity": rnd(tp / max(tp + fn, 1)), "specificity": rnd(tn / max(tn + fp, 1)),
                        "kappa": rnd(cohens_kappa(r1.astype(int), p1.astype(int))),
                        "ci_kappa": ci(boot_pairs(rows, lambda a, b: cohens_kappa((a >= 1).astype(int), (b >= 1).astype(int))))}
    rb, pb = np.array([burden(v) for v in ref]), np.array([burden(v) for v in pred])
    out["burden_0_1to4_ge5"] = {"confusion_rows_ref_cols_model": confusion(rb, pb, BURDEN).tolist(),
                                 "weighted_kappa": rnd(weighted_kappa(rb, pb, BURDEN_W)),
                                 "ci_kappa": ci(boot_pairs(rows, lambda a, b: weighted_kappa(
                                     np.array([burden(v) for v in a]), np.array([burden(v) for v in b]), BURDEN_W)))}
    return out


def _wk_from(cm_list):
    pass


def _dist_of(seq):
    d = {}
    for x in seq:
        d[x] = d.get(x, 0) + 1
    return d


def _model_mars(c, r):
    t = r["tissue"]
    if t in set(CEREBELLUM):
        return "cerebellar"
    if t in set(BRAINSTEM):
        return "brainstem"
    if t in set(DEEP_GRAY):
        return "deep"
    if t in set(CORTEX) or t in set(WM):
        return _wm_class(c, r)
    return "other"


def _wm_class(cohort, row):
    """MARS distance rule for a WM-majority candidate (deterministic grid lookup, same code as the reference side)."""
    try:
        cid = row["case"]
        img = nib.load(f"{GRID_DIR[cohort](cid)}/{cid}_image.nii.gz")
        p = SEG_PATH[cohort](cid)
        if not os.path.exists(p):
            return "other"
        seg_full = load_synthseg_like(p, img)
        cx, cy, cz = row["centre_voxel"]
        box = tuple(slice(max(a - 3, 0), a + 4) for a in (cx, cy, cz))
        seg = seg_full[box]
        mask = np.zeros(seg.shape, bool)
        mask[cx - box[0].start, cy - box[1].start, cz - box[2].start] = True
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
        r = classify_lesion(seg, mask, spacing)
        return r["mars"] if r["mars"] in ("deep", "lobar") else "other"
    except Exception:
        return "other"


if __name__ == "__main__":
    main()
