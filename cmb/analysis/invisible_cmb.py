"""QC of the "invisible" reference microbleeds (Arena 2, wc10, 2026-11-05).

With the frozen stage 1 (a03-aniso-e60) about a quarter of the reference CMB of the
hospital cohorts get stage-1 probability EXACTLY 0 (uint8 < 1/255). This script asks,
per reference lesion (26-connected component of the label), WHY the network is blind
to it. For every lesion of the three cohorts it computes:

  visible       max stage-1 probability inside >= 0.01 (else "invisible")
  size          voxels and mm^3
  slices        number of z-slices the lesion touches
  region        SynthSeg region group of the majority label under the lesion
                (lobar = cortical + juxtacortical white matter, deep_grey,
                deep_white, infratentorial = brainstem + cerebellum, csf_other)
  contrast      mean image inside / mean of a 2-voxel shell around the lesion
  darkest_mm    distance (mm) of the darkest image voxel in a 5-voxel window around
                the lesion centroid to the label centroid (label-shift check)
  edge_mm       distance (mm) from the lesion centroid to the brain/head edge
                (first non-brain voxel along the nearest axis)
  edge_slices   lesion touches the first or last 2 slices of the volume (yes/no)
  fold

Private per-lesion table  -> MB/stage2/schleife/invisible_cmb_lesions.tsv   (never public)
Aggregate per cohort      -> MB/stage2/schleife/invisible_cmb_summary.json

CPU only. Frozen folds/models unchanged. Inputs exactly as in cmb.analysis.recall_curve:
  valdo   predictions ergebnisse/turnier/a03-aniso-e60-gd, grid dev/gitter_d
  t2star  predictions <private>/transfer/a03-aniso-e60-cat-t2s,
          manifest MB/gitter/manifest_t2star.json
  swi     predictions <private>/transfer/a03-aniso-e60-cat-swi,
          manifest MB/gitter/manifest_swi_mix.json

    python -m cmb.analysis.invisible_cmb          (writes the TSV + JSON, prints summary)
    python cmb/analysis/invisible_cmb.py --selftest   (synthetic volume, exit 0/1)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage

from cmb.paths import ROOT
from cmb.paths import MB, ID_PREFIX
OUT_TSV = f"{MB}/stage2/schleife/invisible_cmb_lesions.tsv"
OUT_JSON = f"{MB}/stage2/schleife/invisible_cmb_summary.json"
VOXEL_MM3 = 0.25          # arena grid: 0.5 x 0.5 x 1.0 mm
C26 = np.ones((3, 3, 3), bool)
VISIBLE_CUTOFF = 0.01

# SynthSeg label -> region group (FreeSurfer label table, as in code/gewebekarte.py)
CSF_L = {4, 5, 14, 15, 24, 43, 44}
LOBAR_L = {2, 3, 41, 42, 17, 18, 53, 54}        # cortex + juxtacortical WM/hippocampus
DEEP_GREY_L = {10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60}
INFRAT_L = {7, 8, 16, 46, 47}                   # cerebellum GM/WM + brainstem


def region_of(label: int) -> str:
    if label in CSF_L:
        return "csf_other"
    if label in DEEP_GREY_L:
        return "deep_grey"
    if label in INFRAT_L:
        return "infratentorial"
    if label in LOBAR_L:
        return "lobar"
    # remaining white matter labels (2,41 already lobar) -> deep white
    return "deep_white"


def region_majority(seg: np.ndarray, mask: np.ndarray) -> str:
    """Majority SynthSeg region under the lesion; if fully background, dilate by 1."""
    for dil in (0, 1):
        m = mask if dil == 0 else ndimage.binary_dilation(mask, iterations=1)
        counts = Counter(region_of(int(l)) for l in np.unique(seg[m]) if l != 0)
        if counts:
            return counts.most_common(1)[0][0]
    return "csf_other"


def brain_mask(seg: np.ndarray) -> np.ndarray:
    return seg != 0


def edge_transform(seg):
    """Distance transform of the brain mask (voxels to the nearest background),
    with a 2-voxel background pad so the outer volume border is a real edge."""
    brain = np.pad(brain_mask(seg), 2, constant_values=0)
    return ndimage.distance_transform_edt(brain), brain


def edge_distance_mm(centroid, dt, brain, spacing):
    c = np.round(np.asarray(centroid, float)).astype(int) + 2
    if not brain[tuple(int(v) for v in c)]:        # centroid sits outside the brain: 0
        return 0.0
    return float(dt[tuple(int(v) for v in c)]) * float(np.mean(spacing))


def lesion_features(img, frst, prob, lab, seg, spacing):
    """Per-lesion rows for one case. img = raw image, frst = inverted FRST input,
    prob = stage-1 probability, lab = 26-labelled reference, seg = SynthSeg (resampled)."""
    rows = []
    zmax = img.shape[2]
    dt, brain = edge_transform(seg)               # once per case, not per lesion
    for j in range(1, lab.max() + 1):
        m = lab == j
        vox = int(m.sum())
        idx = np.argwhere(m)
        centroid = idx.mean(axis=0)
        pmax = float(prob[m].max())
        # contrast on the FRST (network input) scale: mean inside / mean 2-voxel shell
        # (dilation on the tight bounding box, not the whole volume)
        lo = np.clip(np.round(centroid).astype(int) - 4, 0, None)
        hi = np.minimum(np.round(centroid).astype(int) + 5, img.shape)
        box_m = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        box_f = frst[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        box_dil = ndimage.binary_dilation(box_m, iterations=2)
        shell = box_dil & ~box_m
        inside = float(box_f[box_m].mean())
        outside = float(box_f[shell].mean()) if shell.any() else float("nan")
        contrast = inside / outside if outside and outside > 0 else float("nan")
        # darkest voxel of the RAW image == brightest voxel on the inverted FRST scale
        dk = np.unravel_index(np.argmax(box_f), box_f.shape)
        dk = np.asarray(dk, float) + np.asarray(lo, float)
        off = float(np.sqrt(((dk - centroid) * np.asarray(spacing)) ** 2).sum())
        edge_slices = bool(idx[:, 2].min() < 2 or idx[:, 2].max() >= zmax - 2)
        rows.append(dict(
            cx=int(round(centroid[0])), cy=int(round(centroid[1])), cz=int(round(centroid[2])),
            n_voxels=vox, size_mm3=round(vox * VOXEL_MM3, 2),
            n_slices=int(idx[:, 2].max() - idx[:, 2].min() + 1),
            region=region_majority(seg, m),
            contrast=round(contrast, 4) if contrast == contrast else None,
            darkest_offset_mm=round(off, 2),
            edge_mm=round(edge_distance_mm(centroid, dt, brain, spacing), 2),
            edge_slices=edge_slices,
            p_max=round(pmax, 4),
            visible=pmax >= VISIBLE_CUTOFF,
        ))
    return rows


def load_case(case, pred_dir, image_dir, synthseg_path, fold, cohort):
    ppath = None
    for p in glob.glob(f"{pred_dir}/fold*/{case}_pred_proba.nii.gz"):
        ppath = p
        break
    if ppath is None:
        return None
    prob = np.asarray(nib.load(ppath).dataobj).astype(np.float32) / 255.0
    img_ni = nib.load(f"{image_dir}/{case}_image.nii.gz")
    img = np.asarray(img_ni.dataobj, dtype=np.float32)
    frst = np.asarray(nib.load(f"{image_dir}/{case}_frst.nii.gz").dataobj, dtype=np.float32)
    ref, n = ndimage.label(np.asarray(nib.load(f"{image_dir}/{case}_label.nii.gz").dataobj) > 0,
                           structure=C26)
    seg = None
    if synthseg_path and os.path.exists(synthseg_path):
        s = nib.load(synthseg_path)
        seg = np.asarray(resample_from_to(s, img_ni, order=0).dataobj).astype(np.int32) \
            if s.shape != img_ni.shape else np.asarray(s.dataobj).astype(np.int32)
    spacing = np.asarray(img_ni.header.get_zooms()[:3], float)
    rows = lesion_features(img, frst, prob, ref, seg if seg is not None else np.zeros(img.shape, int), spacing)
    for r in rows:
        r.update(cohort=cohort, case=case, fold=fold,
                 image_dir=image_dir, pred_dir=pred_dir)
    return rows


def cases_valdo():
    pred = f"{ROOT}/ergebnisse/turnier/a03-aniso-e60-gd"
    for path in sorted(glob.glob(f"{pred}/fold*/*_pred_proba.nii.gz")):
        case = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
        fold = int(os.path.basename(os.path.dirname(path))[4:])
        yield case, pred, f"{ROOT}/dev/gitter_d/{case}", f"{ROOT}/dev/synthseg/{case}_synthseg.nii.gz", fold, "valdo"


def cases_manifest(pred_dir, manifest, synth_dir, strip, cohort):
    folders = {e["id"]: e for e in json.load(open(manifest))}
    for path in sorted(glob.glob(f"{pred_dir}/fold*/*_pred_proba.nii.gz")):
        case = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
        if case not in folders:
            continue
        e = folders[case]
        fold = int(e["fold"])
        yield case, pred_dir, e["dir"], f"{synth_dir}/robust/{case.replace(strip, '')}_synthseg.nii.gz", fold, cohort


def aggregate(rows, cohort):
    inv = [r for r in rows if not r["visible"]]
    vis = [r for r in rows if r["visible"]]

    def med(vals):
        vals = np.array([v for v in vals if v is not None and v == v], float)
        if not len(vals):
            return [None, None, None]
        return [round(float(np.median(vals)), 3), round(float(np.percentile(vals, 25)), 3),
                round(float(np.percentile(vals, 75)), 3)]

    def share(vals, pred):
        vals = [v for v in vals if v is not None and v == v]
        return round(float(np.mean([pred(v) for v in vals])), 3) if vals else None

    def block(g):
        return dict(
            n=len(g),
            size_mm3=med([r["size_mm3"] for r in g]),
            n_slices=med([r["n_slices"] for r in g]),
            contrast=med([r["contrast"] for r in g]),
            darkest_offset_mm=med([r["darkest_offset_mm"] for r in g]),
            edge_mm=med([r["edge_mm"] for r in g]),
            region={k: round(c / max(len(g), 1), 3) for k, c in
                    Counter(r["region"] for r in g).most_common()},
            edge_slices_share=share([r["edge_slices"] for r in g], lambda v: float(v)),
        )

    return dict(n_invisible=len(inv), n_visible=len(vis), invisible=block(inv), visible=block(vis))


def run(verbose=True):
    t0 = time.time()
    all_rows = []
    for name, gen in (
        ("valdo", cases_valdo()),
        ("t2star", cases_manifest(f"{MB}/transfer/a03-aniso-e60-cat-t2s",
                                  f"{MB}/gitter/manifest_t2star.json",
                                  f"{MB}/synthseg-t2star", ID_PREFIX["t2star"] + "-", "t2star")),
        ("swi", cases_manifest(f"{MB}/transfer/a03-aniso-e60-cat-swi",
                               f"{MB}/gitter/manifest_swi_mix.json",
                               f"{MB}/synthseg-swi", ID_PREFIX["swi"] + "-", "swi")),
    ):
        n = 0
        for case, pred_dir, image_dir, seg_p, fold, cohort in gen:
            try:
                rows = load_case(case, pred_dir, image_dir, seg_p, fold, cohort)
            except Exception as exc:                 # one broken case must not kill the run
                if verbose:
                    print(f"  {name} {case}: FAILED ({exc!r})", flush=True)
                continue
            all_rows += rows
            n += len(rows)
            if verbose and n % 50 < len(rows):
                print(f"  {name}: {n} lesions ...", flush=True)
    cols = ["cohort", "case", "fold", "visible", "cx", "cy", "cz", "n_voxels", "size_mm3",
            "n_slices", "region", "contrast", "darkest_offset_mm", "edge_mm", "edge_slices",
            "p_max", "image_dir", "pred_dir"]
    with open(OUT_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in all_rows:
            f.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    summary = {name: aggregate([r for r in all_rows if r["cohort"] == name], name)
               for name in ("valdo", "t2star", "swi")}
    summary["meta"] = dict(
        cutoff=VISIBLE_CUTOFF, voxel_mm3=VOXEL_MM3,
        n_lesions={c: len([r for r in all_rows if r["cohort"] == c]) for c in ("valdo", "t2star", "swi")},
        seconds=round(time.time() - t0, 1),
        date="2026-11-05",
    )
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=1)
    if verbose:
        for c in ("valdo", "t2star", "swi"):
            s = summary[c]
            print(f"{c}: {s['n_invisible']} invisible / {s['n_visible']} visible; "
                  f"contrast med {s['invisible']['contrast']} (inv) vs {s['visible']['contrast']} (vis); "
                  f"size {s['invisible']['size_mm3']} vs {s['visible']['size_mm3']}; "
                  f"edge_slices {s['invisible']['edge_slices_share']} vs {s['visible']['edge_slices_share']}",
                  flush=True)
    return summary


def selftest():
    """Synthetic volume: one dark blob with stage-1 probability 0, one with 0.9.
    The p=0 blob must be invisible, the p=0.9 one visible. Exit 0 on success."""
    rng = np.random.default_rng(0)
    shape = (40, 40, 30)
    img = (0.5 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
    frst = (0.5 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
    prob = np.zeros(shape, np.float32)
    # brain: an ellipsoid, label 3 (cortex) everywhere, so region + edge work
    z, y, x = np.indices(shape)
    brain = ((x - 20) ** 2 / 30 ** 2 + (y - 20) ** 2 / 30 ** 2 + (z - 15) ** 2 / 12 ** 2) < 1.0
    seg = np.where(brain, 3, 0).astype(np.int32)
    # lesion A (invisible): dark blob, probability 0
    a = (x - 12) ** 2 + (y - 14) ** 2 + (z - 10) ** 2 < 9
    # lesion B (visible): dark blob, probability 0.9
    b = (x - 28) ** 2 + (y - 26) ** 2 + (z - 20) ** 2 < 9
    lab = np.zeros(shape, int)
    lab[a] = 1
    lab[b] = 2
    img[a] = 0.2; img[b] = 0.2
    frst[a] = 0.9; frst[b] = 0.9          # on FRST scale lesions are bright
    prob[a] = 0.0
    prob[b] = 0.9
    rows = lesion_features(img, frst, prob, lab, seg, np.array([0.5, 0.5, 1.0]))
    assert len(rows) == 2, f"expected 2 lesions, got {len(rows)}"
    rA = next(r for r in rows if r["p_max"] == 0.0)
    rB = next(r for r in rows if r["p_max"] == 0.9)
    assert rA["visible"] is False, "p=0 lesion must be invisible"
    assert rB["visible"] is True, "p=0.9 lesion must be visible"
    assert rA["region"] == "lobar", rA
    assert 0 < rA["edge_mm"] < 50, rA
    assert rA["n_slices"] >= 1 and rA["n_voxels"] >= 1
    assert rA["contrast"] is not None and rA["contrast"] > 1, rA
    print("selftest OK:", {k: rA[k] for k in ("visible", "region", "contrast", "edge_mm", "darkest_offset_mm")})
    print("selftest OK:", {k: rB[k] for k in ("visible", "region", "contrast", "edge_mm", "darkest_offset_mm")})
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    run()
