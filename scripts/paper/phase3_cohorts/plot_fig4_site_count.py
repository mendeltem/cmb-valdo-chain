#!/usr/bin/env python3
"""wk3 fig4: site-count curves, tset-plain (HPC redo 06.11.2026).

Two curves per held-out site:
  growing size  -- stage 2 trained on ALL candidates of k other sites (phase3c)
  fixed size    -- stage 2 trained on a FIXED pool N_fix for every k (phase3c_fixed)
Question: does each added site / data type improve the generalist classifier,
and does fixed-size rise with k (diversity beyond data size)?

Input : <private>/paper/phase3_tset_hpc/phase3c_summary.json
        <private>/paper/phase3_tset_hpc/phase3c_fixed_summary.json
Output: <repo>/paper/fig4_site_count_curve.png  (overwrites the forest figure)
        <private>/paper/fig4_site_count_curve.png   (copy)
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

HPC = _P + "/paper/phase3_tset_hpc"
SITES = ["ALFA", "RSS", "SABRE", "swi", "t2star"]
K = [1, 2, 3, 4]

grow = json.load(open(f"{HPC}/phase3c_summary.json"))
fix = json.load(open(f"{HPC}/phase3c_fixed_summary.json"))
assert grow["meta"]["model"] == "tset-plain"
assert fix["meta"]["model"] == "tset-plain"

def curve(d, site):
    ks, fs, lo, hi = [], [], [], []
    for k in K:
        v = d[site].get(str(k))
        if not v:
            continue
        ks.append(k)
        fs.append(v["f1"])
        lo.append(v["f1"] - (v["f1"] - v["ci95"][0]))
        hi.append(v["f1"] + (v["ci95"][1] - v["f1"]))
    return ks, fs, lo, hi

# pooled: mean of the per-site curves at each k (equal weight per site)
def pooled(d, site):
    ks, fs, lo, hi = curve(d, site)
    return ks, fs

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()

# panel 0: pooled (mean over sites)
pg = {k: [] for k in K}
pf = {k: [] for k in K}
for s in SITES:
    gks, gfs, _, _ = curve(grow, s)
    fks, ffs, _, _ = curve(fix, s)
    for k, v in zip(gks, gfs):
        pg[k].append(v)
    for k, v in zip(fks, ffs):
        pf[k].append(v)
ax = axes[0]
ax.plot([k for k in K if pg[k]], [sum(pg[k])/len(pg[k]) for k in K if pg[k]],
        "o-", color="tab:blue", label="growing size")
ax.plot([k for k in K if pf[k]], [sum(pf[k])/len(pf[k]) for k in K if pf[k]],
        "s--", color="tab:red", label="fixed size (N_fix)")
ax.set_title("pooled (mean over 5 sites)")
ax.set_xlabel("k = # training sites"); ax.set_ylabel("F1 on held-out site")
ax.legend(); ax.grid(alpha=0.3)

# one panel per site
for i, s in enumerate(SITES, start=1):
    ax = axes[i]
    gks, gfs, glo, ghi = curve(grow, s)
    fks, ffs, flo, fhi = curve(fix, s)
    ax.plot(gks, gfs, "o-", color="tab:blue", label="growing size")
    ax.fill_between(gks, glo, ghi, color="tab:blue", alpha=0.15)
    ax.plot(fks, ffs, "s--", color="tab:red", label="fixed size (N_fix=%d)" % fix[s].get("N_fix", "?"))
    ax.fill_between(fks, flo, fhi, color="tab:red", alpha=0.15)
    ax.set_title(s)
    ax.set_xlabel("k = # training sites"); ax.set_ylabel("F1 on held-out site")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# last panel: slope text (bootstrap dF1/dk; only the fixed-size summary
# carries slope_test in the HPC merge)
ax = axes[5]
ax.axis("off")
fslope = fix["slope_test"]["per_site"]
lines = ["slope test (bootstrap, dF1/dk)",
         "fixed-size curve:", ""]
for s in SITES:
    f = fslope.get(s, {})
    ci = f.get("ci95", [float("nan")] * 2)
    lines.append("%-7s %+0.4f  CI95 [%+0.4f, %+0.4f]" % (
        s, f.get("slope_per_site", float("nan")), ci[0], ci[1]))
fp = fix["slope_test"].get("pooled", {})
cip = fp.get("ci95", [float("nan")] * 2)
lines += ["", "pooled  %+0.4f  CI95 [%+0.4f, %+0.4f]" % (
    fp.get("slope_per_site", float("nan")), cip[0], cip[1])]
lines += ["", "(slope_test not in phase3c merge;"]
lines.append(" growing curve plotted without test.)")
ax.text(0.02, 0.98, "\n".join(lines), family="monospace", va="top")
ax.set_title("slope test")

fig.suptitle("wk3 phase III site-count curve, tset-plain (HPC, 06.11.2026)", y=0.995)
fig.tight_layout()
fig.savefig(_R + "/paper/fig4_site_count_curve.png", dpi=130)
fig.savefig(_P + "/paper/fig4_site_count_curve.png", dpi=130)
print("wrote fig4 (growing + fixed size, tset-plain)")
