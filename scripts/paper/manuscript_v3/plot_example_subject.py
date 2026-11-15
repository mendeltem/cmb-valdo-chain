"""Fig. S3 candidate -- three in-house subjects (one row each), axial: T2* | reference | n = 0 | 5 | 10 | 15 | 20 | 30 | 40 (fine-tuned, seed 42).
User request 02.11.2026: "column 1 T2*, column 2 ground truth, then n=0, 5, 10, 15, 20, 30, 40 ... choose the subject whose
numbers show the improvement best, first only one subject".

Selection rule (fixed here, applied by the script; the analyst does not look at patient images in "arbeit" mode):
  test subjects of fold 0 with >= 4 reference CMB; score = F1(n=20) - F1(n=0), minus 0.5 x the largest drop between
  consecutive n in 0, 5, 10, 15, 20 (prefers a steady rise); highest score wins. Ties -> more reference CMB.
Per-case F1 comes from the QC page's data.json (same post-processing as the learning curve).
Row 1: whole axial slice (slice with the most reference CMB). Row 2: zoom (64 x 64 voxels) around the densest CMB group.
Rings: green = reference CMB found, orange = false positive, blue = missed; the titles count the WHOLE scan.

    python plot_example_subject.py [--cohort t2star] [--case <id>]    # --case overrides the rule
Writes figures/main/Fig3_examples_mri.png (T2*; other cohorts: figures/supplement/FigS3x_example_<cohort>.png) and prints only the rule's outcome (no values per case).
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
import argparse, json, os, sys
import numpy as np, nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QC = _os.environ.get("CMB_QC_PLOTS", _P + "/qc_plots")
COLS = ["n=0 (VALDO model as downloaded)", "FT n=5", "FT n=10", "FT n=15", "FT n=20", "FT n=30", "FT n=40"]
LABEL = {"n=0 (VALDO model as downloaded)": "n = 0"} | {c: "n = " + c.split("=")[1] for c in COLS[1:]}
POST = dict(threshold=0.3, min_mm3=2.0, max_voxels=2100)
C26 = np.ones((3, 3, 3), bool)


def f1(st):
    tp, fp, fn = st.get("tp", 0), st.get("fp", 0), st.get("fn", 0)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None


def choose(out_dir):
    d = json.load(open(f"{out_dir}/data.json")); cases = d["cases"] if isinstance(d, dict) else d
    best = None
    for c in cases:
        if (c.get("n_ref") or 0) < 4:
            continue
        st = c.get("stats") or {}
        seq = [f1(st.get(r, {})) for r in COLS[:5]]
        if None in seq:
            continue
        drop = max([0.0] + [seq[i] - seq[i + 1] for i in range(len(seq) - 1)])
        score = (seq[-1] - seq[0]) - 0.5 * drop
        best = (best or []) + [((round(score, 4), c.get("n_ref")), c["id"])]
    return [cid for _, cid in sorted(best or [], reverse=True)]


def post(path, vv):
    p = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
    p = p / 255.0 if p.max() > 1.5 else p
    lab, n = ndimage.label(p > POST["threshold"], C26)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    return np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s * vv >= POST["min_mm3"] and s <= POST["max_voxels"]])


def render_row(axs, case, nr_label):
    nii = nib.load(case["reference"]); vv = abs(float(np.linalg.det(nii.affine[:3, :3])))
    img = np.asanyarray(nib.load(case["image"]).dataobj).astype(np.float32)
    img = 1.0 - img / max(float(img.max()), 1e-6)                  # grids store 1 - I/max: back to CMB dark
    ref = np.asanyarray(nii.dataobj) > 0
    rl, nr = ndimage.label(ref, C26)
    preds = {r: post(case["runs"][r], vv) for r in COLS if r in case["runs"]}
    per_z = np.array([len(np.unique(rl[:, :, z][rl[:, :, z] > 0])) for z in range(ref.shape[2])])
    z = int(np.argmax(per_z))
    sl, rs = np.rot90(img[:, :, z]), np.rot90(ref[:, :, z])
    brain = np.nonzero(sl > 0.02)
    by0, by1, bx0, bx1 = brain[0].min(), brain[0].max() + 1, brain[1].min(), brain[1].max() + 1

    def ring(ax, m2, colour):
        lab2, n2 = ndimage.label(m2)
        for c in ndimage.center_of_mass(m2, lab2, range(1, n2 + 1)):
            ax.add_patch(plt.Circle((c[1] - bx0, c[0] - by0), 7, fill=False, color=colour, lw=1.5))

    cols = ["T2*", "reference"] + [r for r in COLS if r in preds]
    for j, col in enumerate(cols):
        ax = axs[j]; ax.imshow(sl[by0:by1, bx0:bx1], cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        if col == "T2*":
            ax.set_title("T2*", fontsize=10); ax.set_ylabel(nr_label, fontsize=10)
        elif col == "reference":
            ring(ax, rs, "#1f77b4"); ax.set_title(f"reference\n{nr} CMB", fontsize=10)
        else:
            pm = preds[col]; pl, _ = ndimage.label(pm, C26)
            found = np.isin(rl, np.unique(rl[pm & ref])) & ref
            fp = pm & ~np.isin(pl, np.unique(pl[ref & pm]))
            nf = len(set(np.unique(rl[found])) - {0}); nfp = len(set(np.unique(pl[fp])) - {0})
            for m2, colour in ((found, "#2ca02c"), (fp, "#ff7f0e"), (ref & ~found, "#1f77b4")):
                ring(ax, np.rot90(m2[:, :, z]), colour)
            ax.set_title(f"{LABEL[col]}\n{nf}/{nr} found, {nfp} FP", fontsize=10)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", default="t2star"); ap.add_argument("--case", nargs="*")
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--subjects", type=int, default=3); a = ap.parse_args()
    out_dir = f"{QC}/cmb_learning_{a.cohort}_fold{a.fold}"
    man = json.load(open(f"{out_dir}/manifest.json"))
    ids = a.case or choose(out_dir)[: a.subjects]
    if not ids:
        print("no subject meets the rule"); return 1
    ncol = 2 + len(COLS)
    fig, axs = plt.subplots(len(ids), ncol, figsize=(2.05 * ncol, 3.05 * len(ids) + 0.5), facecolor="white", squeeze=False)
    for i, cid in enumerate(ids):
        render_row(axs[i], next(c for c in man["cases"] if c["id"] == cid), f"subject {chr(65 + i)}")
    fig.text(0.01, 0.01, f"in-house {'T2* (5 mm)' if a.cohort == 't2star' else 'SWI'}, test subjects of fold {a.fold} (A-{chr(64 + len(ids))}), "
             "each on its axial slice with the most reference CMB. Fine-tuned from the VALDO model with n local scans (seed 42), "
             "stage 1, fixed post-processing.\nRings: green = reference CMB found, orange = false positive, blue = missed in this "
             "slice; counts above each panel cover the whole scan.", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=2.2)   # 02.11.: two-line titles touched the row above
    out = f"{P}/figures/main/Fig3_examples_mri.png" if a.cohort == "t2star" else f"{P}/figures/supplement/FigS3x_example_{a.cohort}.png"   # 02.11.: main Fig. 3
    fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    print(f"{len(ids)} subjects", "selected by the rule" if not a.case else "given", "-> written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
