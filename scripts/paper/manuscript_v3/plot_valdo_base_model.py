"""The VALDO base model of phase 2 (the start point of every FT block): how it was trained and what it does on its own data
versus on the in-house scans as downloaded (n = 0).

    python plot_valdo_base_model.py

A  training loss per epoch: the two full-data members (72 VALDO scans, 60 epochs, state after the last epoch is kept)
   and the five folds of the VALDO cross-validation with the same recipe (a03-aniso-e60-gd, 57 scans).
B  inner-validation lesion F1 per evaluated epoch in that cross-validation (the full-data members have no validation).
C  lesion F1, sensitivity and precision: VALDO cross-validation test folds (pooled) vs the full-data model on in-house
   T2* and SWI without any in-house training (phase4_eval_*_n0.json); false positives per scan written on the bars.
Writes figures/supplement/FigS2_valdo_base_model.png. Aggregates only.
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
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.dirname(P)
MB = _P
CV = f"{V}/ergebnisse/turnier/a03-aniso-e60-gd"
FULL = f"{V}/ergebnisse/final/valdo_t2"

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), facecolor="white", gridspec_kw=dict(width_ratios=[1, 1, 1.15]))

ax = axs[0]
for k in range(5):
    v = json.load(open(f"{CV}/fold{k}/verlauf.json"))
    ax.plot([x["epoche"] for x in v], [x["loss"] for x in v], color="#95a5a6", lw=1, alpha=0.8,
            label="VALDO cross-validation folds (57 scans)" if k == 0 else None)
for s, col in (("seed42", "#c0392b"), ("seed1", "#e67e22")):
    v = json.load(open(f"{FULL}/{s}/verlauf.json"))
    ax.plot([x["epoch"] for x in v], [x["loss"] for x in v], color=col, lw=2, label=f"Full-data member {s} (72 scans)")
ax.set_title("A  Training loss (BCE + Dice, deep supervision)", loc="left", fontsize=11)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(frameon=False, fontsize=8.5)

ax = axs[1]
cv_tp = cv_fp = cv_fn = cv_cases = 0
for k in range(5):
    v = [x for x in json.load(open(f"{CV}/fold{k}/verlauf.json")) if "innen_f1" in x]
    m = json.load(open(f"{CV}/fold{k}/meta.json"))
    ax.plot([x["epoche"] for x in v], [x["innen_f1"] for x in v], "-o", ms=3, lw=1.2, color=plt.cm.viridis(k / 4.5),
            label=f"fold {k} (test F1 {m['test_summary']['f1']:.2f})")
    e = m["epoche_gewaehlt"]; y = [x["innen_f1"] for x in v if x["epoche"] == e]
    if y:
        ax.scatter([e], y, s=70, facecolors="none", edgecolors="black", zorder=4)
    t = m["test_summary"]; cv_tp += t["tp"]; cv_fp += t["fp"]; cv_fn += t["fn"]; cv_cases += t["cases"]
ax.set_title("B  Inner-validation F1 (circle = chosen epoch)", loc="left", fontsize=11)
ax.set_xlabel("Epoch"); ax.set_ylabel("Lesion F1 (inner fold)"); ax.set_ylim(0, 0.9); ax.legend(frameon=False, fontsize=8, ncol=2)

ax = axs[2]
cv = dict(f1=2 * cv_tp / (2 * cv_tp + cv_fp + cv_fn), sensitivity=cv_tp / (cv_tp + cv_fn), precision=cv_tp / (cv_tp + cv_fp),
          fp_per_case=cv_fp / cv_cases)
t2 = json.load(open(f"{MB}/paper/phase4_eval_t2s_n0.json"))["runs"]["valdo72-on-t2s"][_LABEL + "-t2star"]
# 03.11.2026: SWI n = 0 from the current SWI grid (phase4/calibration_swi_blocks.json, N0 blocks, all 5 test folds
# pooled), the same source as manuscript 3.2; phase4_eval_swi_n0.json (30.10.) is an older run and no longer used.
_b = json.load(open(f"{MB}/paper/phase4/calibration_swi_blocks.json"))
_n0 = [_b[f"N0_n0_fold{k}_s0"] for k in range(5)]
_tp, _fp, _fn, _sc = (sum(x[c] for x in _n0) for c in ("tp", "fp", "fn", "scans"))
sw = dict(cases=_sc, f1=2 * _tp / (2 * _tp + _fp + _fn), sensitivity=_tp / (_tp + _fn), precision=_tp / (_tp + _fp),
          fp_per_case=_fp / _sc)
groups = [(f"VALDO itself\n(CV test, {cv_cases} scans)", cv, "#2c3e50"),
          (f"in-house T2*\n(n = 0, {t2['cases']} scans)", t2, "#c0392b"),
          (f"in-house SWI\n(n = 0, {sw['cases']} scans)", sw, "#8e44ad")]
mets = [("f1", "F1"), ("sensitivity", "Sensitivity"), ("precision", "Precision")]
w = 0.26
for i, (lab, d, col) in enumerate(groups):
    for j, (k, _) in enumerate(mets):
        b = ax.bar(j + (i - 1) * w, d[k], w * 0.92, color=col, alpha=[1, 0.75, 0.55][j] if False else 1,
                   label=lab.replace("\n", " ") if j == 0 else None)
        ax.text(j + (i - 1) * w, d[k] + 0.015, f"{d[k]:.2f}", ha="center", fontsize=8)
    ax.text(2.62, 0.93 - 0.075 * i, f"{lab.splitlines()[0]}: {d['fp_per_case']:.1f} FP/scan", color=col, fontsize=8.5, ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in mets]); ax.set_ylim(0, 1)
ax.set_title("C  Own data vs in-house as downloaded", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
for a in axs:
    a.grid(True, color="#e8e8e8", lw=0.8); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.text(0.01, 0.01, "Base model = a03-aniso 3D U-Net (22.5 M parameters), input T2* + FRST, 800 patches per epoch, lr 3e-4, "
         "60 epochs, trained on all 72 public VALDO scans (236 CMB; 3 sites: Philips 3 T, GE 1.5 T, GE 3 T).\nCV and n = 0 values: "
         "stage 1 only; CV threshold chosen on the inner fold, n = 0 fixed 0.3 / 2 mm³.", fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.07, 1, 1))
out = f"{P}/figures/supplement/FigS2_valdo_base_model.png"
fig.savefig(out, dpi=200, facecolor="white"); print(out)
print(json.dumps({k: round(v, 3) for k, v in cv.items()}))
