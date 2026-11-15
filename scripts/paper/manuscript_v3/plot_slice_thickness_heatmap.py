"""Fig. 5 -- slice thickness in the VALDO cross-validation as two heatmaps (descriptive observation, public data).
Full two-stage chain (stage-2 'pooled tset-plain', fixed post-processing), mean of the two stage-2 seeds.

    python plot_slice_thickness_heatmap.py

A  share of reference CMB found, per slice-thickness group (rows) and reference CMB size (columns; equivalent-sphere
   diameter of the 26-connected reference component, 0.25 mm3 voxels of the working grid).
B  share of detections that were wrong, per slice-thickness group and detection size (same size classes); colour and
   big number are this share, the small text adds counts and false positives per scan. Fewer than 3 detections: grey.
Grouping = native slice SPACING (info.json 'quelle_spacing'); SABRE has 3 mm slices at 4 mm spacing (1 mm gaps), labelled so. Aggregates only.
Writes figures/main/Fig5_slice_thickness_size.png (the earlier dot-plot version: plot_slice_thickness.py).
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
V = os.path.dirname(P)
MANIFEST = f"{V}/dev/manifest_valdo_gitter_d.json"
DECISIONS = _P + "/paper/decisions/chainA_s{}.jsonl"
MODE, SEEDS, VOXEL_MM3 = "pooled tset-plain", (0, 1), 0.25
SITE = {"1": "SABRE", "2": "RSS", "3": "ALFA"}
GROUPS = [(0.8, "RSS"), (3.0, "ALFA"), (4.0, "ALFA"), (4.0, "SABRE")]
SIZES = [("<2", "< 2 mm"), ("2-5", "2-5 mm"), (">5", "> 5 mm")]
GOOD_BAD = LinearSegmentedColormap.from_list("good_bad", ["#e74c3c", "#f5f5dc", "#27ae60"])
EMPTY = "#ecf0f1"
MIN_DETECTIONS = 3


def size_class(mm3):
    d = (6 * mm3 / np.pi) ** (1 / 3)
    return "<2" if d < 2 else ("2-5" if d <= 5 else ">5")


group, ref_class = {}, {}
for m in json.load(open(MANIFEST)):
    th = round(json.load(open(f"{m['dir']}/info.json"))["quelle_spacing"][2], 1)
    group[m["id"]] = (th, SITE[m["cohort"][-1]])
    lab, _ = ndimage.label(np.asanyarray(nib.load(f"{m['dir']}/{m['id']}_label.nii.gz").dataobj) > 0, structure=np.ones((3, 3, 3)))
    for i, c in enumerate(np.bincount(lab.ravel())[1:], 1):
        ref_class[(m["id"], i)] = size_class(c * VOXEL_MM3)
scans = collections.Counter(group.values())

found_n, ref_n, det_true, det_false = (collections.Counter() for _ in range(4))
for s in SEEDS:
    found = set()
    for line in open(DECISIONS.format(s)):
        r = json.loads(line)
        if r["cohort"] != "valdo" or r["mode"] != MODE or not r["accepted"]:
            continue
        found.update((r["case"], t) for t in r["touched"])
        key = (group[r["case"]], size_class(r["voxels"] * VOXEL_MM3))
        (det_true if r["touched"] else det_false)[key] += 1
    for k, c in ref_class.items():
        ref_n[(group[k[0]], c)] += 1; found_n[(group[k[0]], c)] += k in found
K = len(SEEDS)


def grid(ax, title, cell):
    for i, g in enumerate(GROUPS):
        for j, (k, _) in enumerate(SIZES):
            colour, big, small = cell(g, k)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colour, ec="white", lw=3))
            if big:
                ax.text(j + .5, i + (.36 if "\n" in small else .42), big, ha="center", va="center", fontsize=15, weight="bold", color="#2c3e50")
                ax.text(j + .5, i + (.8 if "\n" in small else .76), small, ha="center", va="center", fontsize=7.5, color="#2c3e50")
            else:
                ax.text(j + .5, i + .5, small, ha="center", va="center", fontsize=7.5, color="#95a5a6")
    ax.set_xlim(0, len(SIZES)); ax.set_ylim(len(GROUPS), 0)
    ax.set_xticks(np.arange(len(SIZES)) + .5, [n for _, n in SIZES]); ax.xaxis.tick_top(); ax.tick_params(length=0)
    lab = {(4.0, "SABRE"): "3 mm, 1 mm gaps  SABRE"}   # 02.11.: SABRE = 3 mm slices at 4 mm spacing (Sudre 2024); header shows spacing
    ax.set_yticks(np.arange(len(GROUPS)) + .5, [f"{lab.get((t, s), f'{t:g} mm  {s}')}\n({scans[(t, s)]} scans)" for t, s in GROUPS])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, pad=28)


def sensitivity_cell(g, k):
    n = ref_n[(g, k)] // K
    if not n:
        return EMPTY, None, "none visible\nor annotated"
    f = found_n[(g, k)] / K; p = f / n
    return GOOD_BAD(p), f"{p:.0%}", f"{f:.0f} of {n} found"


def false_positive_cell(g, k):
    fp, tp = det_false[(g, k)] / K, det_true[(g, k)] / K
    if fp + tp == 0:
        return EMPTY, None, "no detections"
    if fp + tp < MIN_DETECTIONS:                    # a share from 1-2 detections means nothing: grey, numbers only
        return EMPTY, None, f"too few detections ({fp + tp:g})\n{fp / scans[g]:.2f} FP per scan"
    wrong = fp / (fp + tp)
    return GOOD_BAD(1 - wrong), f"{wrong:.0%} wrong", f"{fp:g} of {fp + tp:g} detections\n{fp / scans[g]:.2f} FP per scan"


fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), facecolor="white")
grid(axs[0], "A  Microbleeds found (sensitivity), by microbleed size", sensitivity_cell)
grid(axs[1], "B  Detections that were wrong (false positives), by detection size", false_positive_cell)
axs[1].set_yticklabels([])
fig.text(0.01, 0.015, "VALDO cross-validation, full two-stage detector, fixed post-processing, mean of two stage-2 seeds. "
         "Size = equivalent-sphere diameter. FP = false positive (wrong detection).\nColour: green good, red poor, in both panels; grey = no or fewer than 3 detections. "
         "Observation on few CMB per cell; thickness is confounded with site and scanner.", fontsize=7.5, color="#7f8c8d")
fig.tight_layout(rect=(0, 0.07, 1, 1))
out = os.environ.get("OUT", f"{P}/figures/main/Fig5_slice_thickness_size.png")
fig.savefig(out, dpi=200, facecolor="white"); print(out)
