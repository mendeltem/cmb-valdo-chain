"""Fig. S5 -- the slice-thickness heatmap of Fig. 5, for the two in-house cohorts: one row per sequence (thickness is uniform
within each cohort: T2* GRE 5 mm in 74 of 75 scans, SWI 2 mm in 52 of 61, 1.5-1.8 mm in the rest), same two panels.
User request 02.11.2026. Full two-stage chain (stage 2 'pooled tset-plain', fixed post-processing, the same out-of-fold
decisions as the clinical analysis), mean of the two stage-2 seeds. Aggregates only.

    python plot_inhouse_size_heatmap.py      # -> figures/supplement/FigS5_inhouse_size_detection.png

A  share of reference CMB found per CMB size (equivalent-sphere diameter of the 26-connected reference component; voxel
   volume from each scan's header). B  share of detections that were wrong, per detection size; fewer than 3: grey.
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
import collections, json, os
import numpy as np, nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MB = _P
MANIFEST = {"t2star": f"{MB}/gitter/manifest_t2star.json", "swi": f"{MB}/gitter/manifest_swi_mix_v2.json"}  # as the dumps
DECISIONS = f"{MB}/paper/decisions/chainA_s{{}}.jsonl"
MODE, SEEDS = "pooled tset-plain", (0, 1)
SIZES = [("<2", "< 2 mm"), ("2-5", "2-5 mm"), (">5", "> 5 mm")]
GOOD_BAD = LinearSegmentedColormap.from_list("good_bad", ["#e74c3c", "#f5f5dc", "#27ae60"])
EMPTY, MIN_DETECTIONS = "#ecf0f1", 3


def size_class(mm3):
    d = (6 * mm3 / np.pi) ** (1 / 3)
    return "<2" if d < 2 else ("2-5" if d <= 5 else ">5")


ref_class, vox_mm3, scans, thick = {}, {}, collections.Counter(), collections.defaultdict(collections.Counter)
for mod, f in MANIFEST.items():
    for e in json.load(open(f)):
        img = nib.load(f"{e['dir']}/{e['id']}_label.nii.gz")
        vv = abs(float(np.linalg.det(img.affine[:3, :3]))); vox_mm3[(mod, e["id"])] = vv
        lab, _ = ndimage.label(np.asanyarray(img.dataobj) > 0, structure=np.ones((3, 3, 3)))
        for i, c in enumerate(np.bincount(lab.ravel())[1:], 1):
            ref_class[(mod, e["id"], i)] = size_class(c * vv)
        scans[mod] += 1
        try:
            thick[mod][round(json.load(open(f"{e['dir']}/info.json"))["quelle_spacing"][2], 1)] += 1
        except (OSError, KeyError, ValueError):
            pass

found_n, ref_n, det_true, det_false = (collections.Counter() for _ in range(4))
for s in SEEDS:
    found = set()
    for line in open(DECISIONS.format(s)):
        r = json.loads(line)
        if r["cohort"] not in MANIFEST or r["mode"] != MODE or not r["accepted"]:
            continue
        key = (r["cohort"], r["case"])
        found.update((r["cohort"], r["case"], t) for t in r["touched"])
        sc = size_class(r["voxels"] * vox_mm3.get(key, 0.25))
        (det_true if r["touched"] else det_false)[(r["cohort"], sc)] += 1
    for k, c in ref_class.items():
        ref_n[(k[0], c)] += 1; found_n[(k[0], c)] += k in found
K = len(SEEDS)
ROWS = [("t2star", "T2* GRE, 5 mm"), ("swi", "SWI, 1.5-2 mm")]


def grid(ax, title, cell):
    for i, (g, _) in enumerate(ROWS):
        for j, (k, _) in enumerate(SIZES):
            colour, big, small = cell(g, k)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colour, ec="white", lw=3))
            if big:
                ax.text(j + .5, i + (.36 if "\n" in small else .42), big, ha="center", va="center", fontsize=15, weight="bold", color="#2c3e50")
                ax.text(j + .5, i + (.8 if "\n" in small else .76), small, ha="center", va="center", fontsize=7.5, color="#2c3e50")
            else:
                ax.text(j + .5, i + .5, small, ha="center", va="center", fontsize=7.5, color="#95a5a6")
    ax.set_xlim(0, len(SIZES)); ax.set_ylim(len(ROWS), 0)
    ax.set_xticks(np.arange(len(SIZES)) + .5, [n for _, n in SIZES]); ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks(np.arange(len(ROWS)) + .5, [f"{lab}\n({scans[g]} scans)" for g, lab in ROWS])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, pad=28)


def sensitivity_cell(g, k):
    n = ref_n[(g, k)] // K
    if not n:
        return EMPTY, None, "none annotated"
    f = found_n[(g, k)] / K; p = f / n
    return GOOD_BAD(p), f"{p:.0%}", f"{f:g} of {n} found"


def false_positive_cell(g, k):
    fp, tp = det_false[(g, k)] / K, det_true[(g, k)] / K
    if fp + tp == 0:
        return EMPTY, None, "no detections"
    if fp + tp < MIN_DETECTIONS:
        return EMPTY, None, f"too few detections ({fp + tp:g})\n{fp / scans[g]:.2f} FP per scan"
    wrong = fp / (fp + tp)
    return GOOD_BAD(1 - wrong), f"{wrong:.0%} wrong", f"{fp:g} of {fp + tp:g} detections\n{fp / scans[g]:.2f} FP per scan"


fig, axs = plt.subplots(1, 2, figsize=(14, 3.6), facecolor="white")
grid(axs[0], "A  Microbleeds found (sensitivity), by microbleed size", sensitivity_cell)
grid(axs[1], "B  Detections that were wrong (false positives), by detection size", false_positive_cell)
axs[1].set_yticklabels([])
fig.text(0.01, 0.02, "in-house cohorts, full two-stage detector after adaptation (out-of-fold, fixed post-processing), mean of two "
         "stage-2 seeds. Size = equivalent-sphere diameter. FP = false positive.\nColour: green good, red poor; grey = none "
         "or fewer than 3 detections. Slice thickness is uniform within each cohort, so one row per sequence.",
         fontsize=7.5, color="#7f8c8d")
fig.tight_layout(rect=(0, 0.1, 1, 1))
out = os.environ.get("OUT", f"{P}/figures/supplement/FigS5_inhouse_size_detection.png")
fig.savefig(out, dpi=200, facecolor="white"); print(out)
print({m: dict(t) for m, t in thick.items()})
