"""Phase-2 learning curve (wk0b): stage-1 lesion metrics against the number of in-house training scans n, for from-scratch (S),
fine-tuned from VALDO (FT) and norm/output-only fine-tuning (FTn). Uses every finished block of every fold, pools the
counts over the test scans of each fold, and plots the mean over seeds (line) and every seed (dots).

    python plot_learning_curve.py [--mod t2star] [--neu]

Post-processing is the fixed stage-1 rule of cmb.transfer.evaluate (threshold, min size, max size, 26-connectivity,
touch matching); the classifier is not applied. Writes figures/main/Fig2_learning_curve_<mod>.png and a cache of the
per-block counts next to this script (only aggregates, no case ids).
"""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import argparse, collections, glob, json, os, sys
import numpy as np

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # paper/
sys.path.insert(0, os.path.dirname(P))                                                    # <repo>/
ARENA = _P
N_ALL = {"t2star": 60, "swi": 49}                       # training scans per fold when n = all (75 - 15, 61 - 12)
N0_F1 = {"t2star": 0.415, "swi": 0.246}                 # VALDO model as downloaded, no fine-tuning (30.10.)


def counts(mod, neu):
    import nibabel as nib
    from scipy import ndimage
    from cmb.transfer.evaluate import THRESHOLD, MIN_MM3, MAX_VOXELS, VOXEL_MM3, CONNECTIVITY_26
    cache_f = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"learning_curve_{mod}.json")
    cache = {} if neu or not os.path.exists(cache_f) else json.load(open(cache_f))
    grid = {"t2star": "manifest_t2star.json", "swi": "manifest_swi_v2.json"}[mod]   # same grids as phase4/run_block.py GRID (02.11.: swi read the old unfilled grid -> shape mismatch)
    man = {e["id"]: e for e in json.load(open(f"{ARENA}/gitter/{grid}"))}
    labels = {}
    for d in sorted(glob.glob(f"{ARENA}/paper/phase4/unet/{mod}/*_fold*_s*")):
        key = os.path.basename(d)
        if key in cache or not os.path.exists(f"{d}/pred_meta.json"):
            continue
        meta = json.load(open(f"{d}/meta.json")); pm = json.load(open(f"{d}/pred_meta.json"))
        fold = key.split("_fold")[1].split("_")[0]
        tp = fp = fn = used = 0
        for c in pm["cases"]:
            f = glob.glob(f"{d}/pred/fold{fold}/{c}_pred_proba.nii.gz")
            if not f:
                continue
            used += 1
            prob = np.asarray(nib.load(f[0]).dataobj).astype(np.float32)
            if prob.max() > 1.5:
                prob /= 255.0
            if c not in labels:
                labels[c] = np.asarray(nib.load(f"{man[c]['dir']}/{c}_label.nii.gz").dataobj) > 0
            lab, n = ndimage.label(prob > THRESHOLD, CONNECTIVITY_26)
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s * VOXEL_MM3 >= MIN_MM3 and s <= MAX_VOXELS])
            pl, npred = ndimage.label(keep, CONNECTIVITY_26); rl, nref = ndimage.label(labels[c], CONNECTIVITY_26)
            hr = len(set(np.unique(rl[pl > 0])) - {0}); hp = len(set(np.unique(pl[rl > 0])) - {0})
            tp += hr; fn += nref - hr; fp += npred - hp
        if used:
            cache[key] = dict(variant=meta["variant"] + ("" if meta.get("strategy", "STRAT") == "STRAT" else "-" + meta["strategy"]), n=str(meta["n"]), seed=meta["seed"], fold=int(fold),
                              tp=tp, fp=fp, fn=fn, scans=used)
            json.dump(cache, open(cache_f, "w"), indent=1)
    return cache


def plot(mod, cache):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nall = N_ALL[mod]
    xn = lambda n: nall if n == "all" else int(n)
    g = collections.defaultdict(list)                     # (variant, n) -> per-seed metrics, counts pooled over folds
    trained = {r["fold"] for r in cache.values() if r["variant"] not in ("N0", "CAL", "CAL-NEG")}
    cache = {k: r for k, r in cache.items() if r["fold"] in trained}     # n = 0 and CAL only on the folds the curves have
    by = collections.defaultdict(lambda: np.zeros(4))
    for r in cache.values():
        by[(r["variant"], r["n"], r["seed"])] += [r["tp"], r["fp"], r["fn"], r["scans"]]
    folds = sorted(trained)
    for (v, n, s), (tp, fp, fn, sc) in by.items():
        g[(v, xn(n))].append(dict(f1=2 * tp / (2 * tp + fp + fn), sens=tp / (tp + fn), prec=tp / max(tp + fp, 1), fps=fp / sc))
    scans = int(max(r["scans"] for r in cache.values())) if len(folds) == 1 else None
    # n = 0 = the VALDO model as is: the start of every curve that begins from it (FT, FTn, CAL, draw strategies)
    if ("N0", 0) in g:
        n0 = g.pop(("N0", 0))[0]; n0_src = "same folds"
    else:
        e = json.load(open(f"{ARENA}/paper/phase4_eval_{'t2s' if mod == 't2star' else mod}_n0.json"))["runs"][f"valdo72-on-{'t2s' if mod == 't2star' else mod}"]
        e = list(e.values())[0]
        n0 = dict(f1=e["f1"], sens=e["sensitivity"], prec=e["precision"], fps=e["fp_per_case"]); n0_src = "all scans"

    style = {"FT": ("#c0392b", "Fine-tuned from VALDO (FT)", "o"),
             "S": ("#2c3e50", "From scratch (S)", "s"),
             "FTn": ("#7f8c8d", "Norm + output layers only (FTn)", "^"),
             "CAL": ("#16a085", "VALDO model, threshold re-chosen on n scans", "P"),
             "CAL-NEG": ("#76d7c4", "VALDO model, threshold from n CMB-free scans (no annotation)", "X"),
             "FT-RAND": ("#e59866", "FT, random draw", "v"),
             "FT-HIGH": ("#922b21", "FT, CMB-richest scans", "D"),
             "FT-LOW": ("#f5b041", "FT, lightest scans (fewest CMB)", "<"),
             "FT-MID": ("#d35400", "FT, median-burden scans", ">"),
             "FT-NONE": ("#5dade2", "FT, CMB-free scans only", "h")}
    panels = [("f1", "Lesion F1"), ("sens", "Sensitivity (recall)"), ("prec", "Precision (PPV)"), ("fps", "False positives per scan")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 9.6), facecolor="white")
    ticks = [2, 3, 5, 10, 15, 20, 30, 40, nall]
    for ax, (m, title) in zip(axs.flat, panels):
        for v, (col, lab, mk) in style.items():
            pts = sorted((n, [x[m] for x in rr]) for (vv, n), rr in g.items() if vv == v)
            if not pts:
                continue
            if v != "S":                                   # from scratch has no n = 0 (an untrained network)
                pts = [(2, [n0[m]])] + pts
            xs = [p[0] for p in pts]
            ax.plot(xs, [np.mean(p[1]) for p in pts], "-", color=col, marker=mk, ms=6, lw=2, label=lab, zorder=3)
            for n, vals in pts:
                ax.scatter([n] * len(vals), vals, color=col, s=14, alpha=0.35, zorder=2, linewidths=0)
        ax.scatter([2], [n0[m]], color="black", marker="D", s=46, zorder=5, facecolors="white", linewidths=1.6,
                   label="n = 0: VALDO model as downloaded" if m == "f1" else None)
        ax.set_xscale("log"); ax.set_xticks(ticks)
        ax.set_xticklabels(["0"] + [str(t) for t in ticks[1:-1]] + [f"all\n({nall})"])
        ax.minorticks_off(); ax.set_xlim(1.8, nall * 1.15)
        ax.set_title(title, fontsize=12, loc="left"); ax.grid(True, color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if m in ("f1", "sens", "prec"):
            vals = [n0[m]] + [x[m] for rr in g.values() for x in rr]
            ax.set_ylim(max(0, min(vals) - 0.05), min(1, max(vals) + 0.05))
        else:
            ax.set_yscale("log"); ax.set_ylabel("log scale", fontsize=8, color="#777")
            from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
            ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 20, 30]))
            ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    for ax in axs[1]:
        ax.set_xlabel("in-house training scans (n, log scale)")
    h, l = axs[0, 0].get_legend_handles_labels()                       # legend below the panels: never over the lines (user rule)
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035))
    where = (f"fold {folds[0]} only, {scans} test scans" if scans else f"{len(folds)} folds pooled")
    name = {"t2star": "T2*", "swi": "SWI"}[mod]
    fig.suptitle(f"Phase 2 learning curve, {name}, stage 1 ({where}; interim)", fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Line = mean over random draws/seeds, dots = single draws. Fixed stage-1 post-processing, "
             f"classifier not applied.\nn = 0 = the unchanged VALDO model ({n0_src}); from scratch starts at n = 3.", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.15, 1, 0.96))
    out = os.path.join(P, "figures", "main", f"Fig2_learning_curve_{mod}.png")
    fig.savefig(out, dpi=200, facecolor="white"); print(out)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--mod", default="t2star"); a.add_argument("--neu", action="store_true")
    a = a.parse_args()
    c = counts(a.mod, a.neu)
    cal = f"{ARENA}/paper/phase4/calibration_{a.mod}_blocks.json"     # calibration-only control (calibrate_n0.py)
    if os.path.exists(cal):
        c = {**c, **json.load(open(cal))}
    plot(a.mod, c)
