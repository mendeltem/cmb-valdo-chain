"""fig5_final.py — final fig5_learning_curve.png (06.11.2026, R5).

Replaces the provisional 30.10 figure (stage-1 points only). Draws the CHAIN
F1(n) curves (stage 2 frozen-V(a), folds pooled, all seeds of the same draw)
per target and per stage-1 variant (S / FT / FTn), with:
  - n = 0: stage-1 only, VALDO U-Net zero-shot + VALDO-only classifier (chain
    n=0 = the public model as is); values from phase4_eval_<mod>_n0.json.
  - n = 'all' (max training-fold size): the full-data point of each variant.
  - reference dashed line at 95 % of the in-site full-data chain F1
    (wk2 S/P control: t2star 0.6672, swi 0.6649).
English, lines never over text (labels in the free corner).
Data: phase4/primary_endpoint.json (computed 06.11.2026 from stored predictions).
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
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P4 = _P + "/paper"
pe = json.load(open(f"{P4}/phase4/primary_endpoint.json"))
n0 = {
    "t2star": json.load(open(f"{P4}/phase4_eval_t2s_n0.json"))["runs"]["valdo72-on-t2s"][_LABEL + "-t2star"]["f1"],
    "swi": json.load(open(f"{P4}/phase4_eval_swi_n0.json"))["runs"]["valdo72-on-swi"][_LABEL + "-swi"]["f1"],
}
ref = pe["reference_chain_f1"]

STY = {
    "S": dict(color="tab:blue", ls="-", marker="o"),
    "FT": dict(color="tab:orange", ls="--", marker="s"),
    "FTn": dict(color="tab:green", ls=":", marker="^"),
}
LABELS = {
    "S": "S: from scratch on site data",
    "FT": "FT: VALDO U-Net fine-tuned",
    "FTn": "FTn: norm + output layers only",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, mod in zip(axes, ("t2star", "swi")):
    for v in ("S", "FT", "FTn"):
        curve = pe["targets"][mod]["chain"][v]["curve"]
        ns = [int(k) for k in curve]
        fs = [curve[str(k)]["f1"] for k in ns]
        x = [0] + ns
        y = [n0[mod]] + fs
        ax.plot(x, y, label=LABELS[v], lw=1.8, ms=5, **STY[v])
    ref95 = 0.95 * ref[mod]
    ax.axhline(ref95, color="k", ls="-.", lw=1.0,
               label=f"95% of our full-data chain F1 ({ref95:.3f})")
    ax.set_xlabel("annotated scans n (site D)")
    ax.set_title(mod)
    ax.set_xlim(-2, max(int(k) for k in list(pe["targets"][mod]["chain"]["S"]["curve"]) + [0]) + 4)
    ax.set_ylim(0, 0.95)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("chain F1 (stage 1 + stage 2 frozen-V, per-case, pooled folds)")
# labels in the free upper-left, lines never over text
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", ncol=2, fontsize=9, framealpha=0.9)
fig.suptitle("How many subjects does a new site need for its CMB U-Net? "
             "(chain F1 vs annotated scans, VALDO public source, frozen 5-fold CV)", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = _R + "/paper/fig5_learning_curve.png"
fig.savefig(out, dpi=150)
print("wrote", out)
