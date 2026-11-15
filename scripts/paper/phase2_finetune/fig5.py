"""fig5: chain F1 vs n per target (S vs FT vs FTn), reference line at 95 % of our
full-data F1. PROVISIONAL (2026-10-30): only the n=0 and n=all, S stage-1 points are
measured; the n-curve trainings are in the GPU queue. The figure is regenerated from the
same script once the curve points exist (phase4_summary.json holds the measured points).

Rules: English, lines never over text.
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

P4 = json.load(open(_P + "/paper/phase4_summary.json"))
mp = P4["measured_points"]

xs = [0, 3, 5, 10, 15, 20, 30, 40]
xlabels = ["0", "3", "5", "10", "15", "20", "30", "40", "all"]


def ref_frac(f1, frac=0.95):
    return frac * f1


fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=150)

# reference: our full-data (in-site, all scans) stage-1 F1 per target
ref = {"t2star": mp["t2star"]["S_all"]["f1"], "swi": mp["swi"]["S_all"]["f1"]}

for mod, lab, c, m in (("t2star", "in-house T2*GRE", "tab:blue", "o"),
                       ("swi", "in-house SWI", "tab:red", "s")):
    y95 = ref_frac(ref[mod])
    ax.axhline(y95, color=c, ls=":", lw=1.0, alpha=0.55)
    ax.plot([0], [mp[mod]["n0"]["f1"]], marker=m, ms=7, color=c, zorder=5,
            label=f"{lab}, n = 0 (VALDO as is)")
    ax.plot([8], [mp[mod]["S_all"]["f1"]], marker=m, ms=7, color=c, zorder=5,
            label=f"{lab}, S at n = all (measured)")
    # measured segment n0 -> S-all (dashed, stage-1 only, no stage-2 chain yet)
    ax.plot([0, 8], [mp[mod]["n0"]["f1"], mp[mod]["S_all"]["f1"]], color=c,
            ls="--", lw=1.2, alpha=0.5, zorder=3)

# 95 % reference annotations (placed where the dashed segments are far away)
ax.text(0.4, ref["swi"] * 0.95 + 0.012, "95 % of SWI full-data F1", color="tab:red",
        fontsize=8, alpha=0.85)
ax.text(0.4, ref["t2star"] * 0.95 + 0.012, "95 % of T2* full-data F1", color="tab:blue",
        fontsize=8, alpha=0.85)

ax.set_xticks(range(9))
ax.set_xticklabels(xlabels)
ax.set_xlabel("n = number of annotated scans in the training fold (nested draws)")
ax.set_ylabel("Stage-1 F1 (lesion-level, fixed post-processing 0.3 / 2 mm$^3$)")
ax.set_title("Provisional: how many subjects does a new site need for its U-Net?\n"
             "(public VALDO source, stage 1 only; full curves after the queued trainings)",
             fontsize=10)
ax.set_ylim(0, 0.85)
ax.grid(alpha=0.25)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

fig.tight_layout()
out = _R + "/paper/fig5_learning_curve.png"
fig.savefig(out)
print("saved", out)
