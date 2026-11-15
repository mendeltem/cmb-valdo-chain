"""wk9 figure: the frozen-V learning curves (measured) next to the A1v2 chain.

Same layout as fig5_learning_curve.png (stage F1 vs n, per modality x variant,
reference line at 95 % of the full-data reference chain F1). If the A1v2 chain
was computed (chains_frozenA1v2.json without all-block errors), its curve is
drawn dashed; otherwise the figure states that A1v2 is pending this round and
shows the frozen-V curves only -- an honest state, not a placeholder number.

Output: <private>/paper/fig_a1v2_learning_curve.png
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
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P4 = _P + "/paper/phase4"
N_ALL = {"t2star": 60, "swi": 49}


def f1_counts(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def load_chain(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def chain_by_n(d, mod, variant):
    out = {}
    for k, v in d.items():
        m = re.match(rf"{mod}/({variant})_n(\d+|all)_fold(\d)_s(\d+)$", k)
        if not m or "error" in v or "a_pooled" not in v:
            continue
        n = N_ALL[mod] if m.group(2) == "all" else int(m.group(2))
        acc = out.setdefault(n, [0, 0, 0])
        for c, (tp_, fp_, fn_) in v["a_pooled"]["per_case"].items():
            acc[0] += tp_; acc[1] += fp_; acc[2] += fn_
    return out


def curve(d, mod, variant):
    c = chain_by_n(d, mod, variant)
    return {n: f1_counts(*t) for n, t in c.items()}


frozenV = load_chain(f"{P4}/chains_frozenV.json")
a1v2 = load_chain(f"{P4}/chains_frozenA1v2.json")
a1v2_ok = False
if a1v2 is not None:
    vals = [v for v in a1v2.values() if "a_pooled" in v]
    a1v2_ok = len(vals) > 0
pe = json.load(open(f"{P4}/primary_endpoint.json"))
ref = {"swi": pe["reference_chain_f1"]["swi"], "t2star": pe["reference_chain_f1"]["t2star"]}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=150, sharey=True)
variants = ("S", "FT", "FTn")
for ax, variant in zip(axes, variants):
    for mod, lab, c, m in (("t2star", "in-house T2*GRE", "tab:blue", "o"),
                           ("swi", "in-house SWI", "tab:red", "s")):
        y95 = 0.95 * ref[mod]
        ax.axhline(y95, color=c, ls=":", lw=1.0, alpha=0.55)
        cv = curve(frozenV, mod, variant)
        if variant == "FTn":
            ns = [n for n in sorted(cv) if n in (5, 10, 20)]
        else:
            ns = sorted(cv)
        ax.plot([n - 0.2 for n in ns], [cv[n] for n in ns], marker=m, ms=6, ls="-",
                color=c, lw=1.6, label=f"frozen-V, {lab}")
        if a1v2_ok:
            ca = curve(a1v2, mod, variant)
            nsA = [n for n in sorted(ca) if n in (5, 10, 20)] if variant == "FTn" else sorted(ca)
            ax.plot([n + 0.2 for n in nsA], [ca[n] for n in nsA], marker=m, ms=6, ls="--",
                    color=c, lw=1.2, alpha=0.8, label=f"A1v2, {lab}")
        else:
            ax.text(0.5, 0.04, "A1v2: pending (target candidate sets not locally available this round)",
                    transform=ax.transAxes, ha="center", fontsize=8, color="0.3",
                    style="italic")
    ax.set_title(variant)
    ax.set_xticks([3, 5, 10, 15, 20, 30, 40, N_ALL["swi"]])
    ax.set_ylim(0, 0.9)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("chain lesion F1, variant (a), folds pooled, all seeds of the draw")
axes[0].set_xlabel("n = annotated scans in the training fold")
axes[1].legend(loc="lower right", fontsize=7)
fig.suptitle("Learning curves: frozen-V (paper chain) and frozen A1v2 (VALDO-only winner)\n"
             "(dotted: 95 % of the full-data reference chain F1)", fontsize=10)
fig.tight_layout()
out = _P + "/paper/fig_a1v2_learning_curve.png"
fig.savefig(out)
print("saved", out)
