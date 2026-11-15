"""Slice thickness in the VALDO cross-validation (descriptive observation, public data): full two-stage chain
(stage-2 'pooled tset-plain', fixed post-processing), both stage-2 seeds pooled.

    python plot_slice_thickness.py

A  lesion sensitivity per slice-thickness group, split by reference CMB size (equivalent-sphere diameter of the
   26-connected reference component, 0.25 mm3 voxels of the working grid); Wilson 95 % intervals, n = CMB per cell.
B  false positives per scan and precision per slice-thickness group (all sizes).
Thickness = native slice spacing (info.json 'quelle_spacing'), site from the VALDO cohort digit.
Writes figures/supplement/FigS3_slice_thickness.png. Aggregates only.
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

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.dirname(P)
MANIFEST = f"{V}/dev/manifest_valdo_gitter_d.json"
DECISIONS = _P + "/paper/decisions/chainA_s{}.jsonl"
MODE, SEEDS, VOXEL_MM3 = "pooled tset-plain", (0, 1), 0.25
SITE = {"1": "SABRE", "2": "RSS", "3": "ALFA"}
GROUPS = [(0.8, "RSS"), (3.0, "ALFA"), (4.0, "ALFA"), (4.0, "SABRE")]
SIZES = [("<2", "< 2 mm", "#c0392b"), ("2-5", "2-5 mm", "#2c3e50"), (">5", "> 5 mm", "#7f8c8d")]


def size_class(mm3):
    d = (6 * mm3 / np.pi) ** (1 / 3)
    return "<2" if d < 2 else ("2-5" if d <= 5 else ">5")


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


group, ref_class = {}, {}
for m in json.load(open(MANIFEST)):
    th = round(json.load(open(f"{m['dir']}/info.json"))["quelle_spacing"][2], 1)
    group[m["id"]] = (th, SITE[m["cohort"][-1]])
    lab, _ = ndimage.label(np.asanyarray(nib.load(f"{m['dir']}/{m['id']}_label.nii.gz").dataobj) > 0, structure=np.ones((3, 3, 3)))
    for i, c in enumerate(np.bincount(lab.ravel())[1:], 1):
        ref_class[(m["id"], i)] = size_class(c * VOXEL_MM3)

hit = collections.Counter(); tot = collections.Counter(); tp = collections.Counter(); fp = collections.Counter()
scans = collections.Counter(g for g in group.values())
for s in SEEDS:
    found = set()
    for line in open(DECISIONS.format(s)):
        r = json.loads(line)
        if r["cohort"] != "valdo" or r["mode"] != MODE or not r["accepted"]:
            continue
        found.update((r["case"], t) for t in r["touched"])
        (tp if r["touched"] else fp)[group[r["case"]]] += 1
    for k, c in ref_class.items():
        tot[(group[k[0]], c)] += 1; hit[(group[k[0]], c)] += k in found

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="white", gridspec_kw=dict(width_ratios=[1.35, 1]))
x = np.arange(len(GROUPS)); labels = [f"{t:g} mm\n{s}" for t, s in GROUPS]

ax = axs[0]
if os.environ.get("X_AXIS") == "size":                      # variant: x = CMB size class, one line per thickness group
    GCOL = ["#27ae60", "#2980b9", "#8e44ad", "#c0392b"]; xs = np.arange(len(SIZES))
    for i, g in enumerate(GROUPS):
        off = (i - 1.5) * 0.12; pts = []
        for j, (key, name, _) in enumerate(SIZES):
            n = tot[(g, key)] // len(SEEDS)
            if n:
                p, lo, hi = wilson(hit[(g, key)] / len(SEEDS), n); pts.append((xs[j] + off, p))
                ax.errorbar(xs[j] + off, p, yerr=[[p - lo], [hi - p]], fmt="o", color=GCOL[i], ms=7, capsize=3, lw=1.3)
                ax.text(xs[j] + off + 0.03, p + 0.025, f"{n}", fontsize=7.5, color=GCOL[i], ha="left")
        ax.plot([], [], "o", color=GCOL[i], ms=7, label=f"{g[0]:g} mm {g[1]}")   # legend only: no connecting lines across other bars
    ax.set_xticks(xs, [n for _, n, _ in SIZES]); ax.set_xlim(-0.5, len(SIZES) - 0.4); ax.set_xlabel("reference CMB size (equivalent diameter); small numbers = n CMB")
    ax.legend(title="slice thickness, site", fontsize=8, title_fontsize=8, loc="upper left", frameon=False, ncol=2)
for j, (key, name, col) in enumerate(SIZES if os.environ.get("X_AXIS") != "size" else []):
    off = (j - 1) * 0.18
    for i, g in enumerate(GROUPS):
        n = tot[(g, key)] // len(SEEDS)
        if n == 0:
            continue
        p, lo, hi = wilson(hit[(g, key)] / len(SEEDS), n)
        ax.errorbar(x[i] + off, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col, ms=7, capsize=3, lw=1.3,
                    label=name if i == next(k for k, gg in enumerate(GROUPS) if tot[(gg, key)]) else None)
        ax.text(x[i] + off + 0.05, p, f"n={n}", fontsize=8, color=col, va="center", ha="left")
if os.environ.get("X_AXIS") != "size":
    ax.set_xticks(x, labels); ax.set_xlim(-0.5, len(GROUPS) - 0.3)
    ax.legend(title="reference CMB size", fontsize=8, title_fontsize=8, loc="upper left", frameon=False, ncol=3)
ax.set_ylim(0, 1.05)
ax.set_ylabel("lesion sensitivity (95 % Wilson interval)")
ax.set_title("A  Sensitivity by CMB size and slice thickness" if os.environ.get("X_AXIS") == "size" else "A  Sensitivity by slice thickness and CMB size", loc="left", fontsize=11)
ax.grid(axis="y", color="#ecf0f1"); ax.set_axisbelow(True)

ax = axs[1]
fps = [fp[g] / (scans[g] * len(SEEDS)) for g in GROUPS]
prec = [tp[g] / max(tp[g] + fp[g], 1) for g in GROUPS]
bars = ax.bar(x, fps, width=0.55, color="#bdc3c7")
for b, v in zip(bars, fps):
    ax.text(b.get_x() + b.get_width() / 2, 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="#2c3e50")
ax.set_ylim(0, 1.6); ax.set_ylabel("false positives per scan (bars)")
ax2 = ax.twinx(); ax2.plot(x, prec, "D", color="#2c3e50", ms=7); ax2.set_ylim(0, 1.05); ax2.set_ylabel("precision (diamonds)")
for i, v in enumerate(prec):
    ax2.text(x[i] + 0.12, v, f"{v:.2f}", fontsize=9, color="#2c3e50", va="center")
ax.set_xticks(x, labels); ax.set_xlim(-0.5, len(GROUPS) - 0.4)
ax.set_title("B  False positives and precision (all sizes)", loc="left", fontsize=11)
for i, g in enumerate(GROUPS):
    ax.text(x[i], 1.52, f"{scans[g]} scans", ha="center", fontsize=8, color="#7f8c8d")

fig.text(0.01, 0.01, "VALDO cross-validation, full two-stage detector, fixed post-processing, mean of two stage-2 seeds. "
         "Observation on few CMB per cell; thickness is confounded with site and scanner.", fontsize=8, color="#7f8c8d")
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = os.environ.get("OUT", f"{P}/figures_src/FigS3_slice_thickness_dotplot_alt.png")   # superseded by Fig. 8 (plot_slice_thickness_heatmap.py)
fig.savefig(out, dpi=200, facecolor="white"); print(out)
