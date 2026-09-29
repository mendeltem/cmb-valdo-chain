"""Robustness probe: which augmentation targets a real weakness of a TRAINED network -- measured at inference, without training.

Why (2026-09-22): the thick-slice augmentation was built on plausibility, cost 1.7 GPU hours and made the network worse
(ARBEIT.md 5p). Before the next training augmentation we want to know, for a few minutes of inference instead of hours of
training, how the finished network reacts when its TEST images are perturbed the way the augmentation would perturb the
training images:

* the F1 collapses at a realistic strength   -> the network is not invariant; training with this perturbation is a candidate;
* the F1 does not move                        -> the network is already invariant; training with it buys nothing in-domain;
* the F1 collapses even at mild strengths that no real scanner produces -> the perturbation destroys the signal; do not train with it.

How: the stored fold models predict their own out-of-fold cases (so everything stays out-of-fold) after the case volumes were
perturbed in memory. Photometric perturbations act on the T2* channel inside the brain only; the radial-symmetry channel is
left alone (it is normalised to relative contrast). Geometric perturbations (in-plane rotation / scaling) are applied to both
channels, the probability map is warped back into the original frame, and the reference is never touched. Scoring is the
project's fixed operating point (threshold 0.3, >= 2 mm3, giant components removed) with optional CSF rule, pooled lesion F1,
and a paired bootstrap over cases against the unperturbed prediction of the same run.

    python -m cmb.analysis.robustness_probe --run ergebnisse/turnier/a03-aniso-e60-gd --net a03-aniso --grid dev/gitter_d \
        --synthseg "dev/synthseg/{id}_synthseg.nii.gz" --json ergebnisse/analysis/robustness_probe_valdo.json

Perturbations and strengths (``--only NAME ...`` restricts; ``--strengths`` is fixed per perturbation, see TABLE):
    gamma     image ** g on the inverted scale (bright = microbleed)     g in 0.7, 0.85, 1.15, 1.5
    bias      smooth multiplicative field exp(N(0, s)) on a 3x3x3 grid    s in 0.05, 0.15, 0.30
    contrast  (image - brain mean) * c + brain mean                       c in 0.7, 0.85, 1.15, 1.3
    noise     + N(0, s * brain std)                                       s in 0.02, 0.05, 0.10
    thick     average over k slices in z, interpolated back (image + FRST) k in 2, 3, 4, 5
    rotate    in-plane rotation by a degrees (linear), prediction rotated back   a in 5, 15
    scale     in-plane zoom by f (linear), prediction zoomed back           f in 0.9, 1.1

GPU: about 30 s per case and condition (57 cases x 22 conditions ~ 6 h on the A5000 if all are run; ``--only`` and ``--cases``
cut that down). CPU works for a smoke test on two cases. Queue it, never start it next to a training.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
sys.path.insert(0, ROOT)
from cmb.analysis.fp_location import load_synthseg, synthseg_path                 # noqa: E402
from cmb.analysis.missed_lesions import majority_label                            # noqa: E402
from cmb.transfer.evaluate import CONNECTIVITY_26, MAX_VOXELS, MIN_MM3, THRESHOLD, VOXEL_MM3   # noqa: E402

CSF_LABELS = {4, 5, 14, 15, 24, 43, 44}
TABLE: Dict[str, List[float]] = {
    "gamma": [0.7, 0.85, 1.15, 1.5],
    "bias": [0.05, 0.15, 0.30],
    "contrast": [0.7, 0.85, 1.15, 1.3],
    "noise": [0.02, 0.05, 0.10],
    "thick": [2, 3, 4, 5],
    "rotate": [5, 15],
    "scale": [0.9, 1.1],
}
IN_PLANE = (1, 2)                 # record volumes are (z, y, x)


# --------------------------------------------------------------------------- perturbations (volumes are (z, y, x), float32)
def photometric(kind: str, strength: float, image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    brain = image != 0
    out = image.copy()
    values = image[brain]
    if kind == "gamma":
        out[brain] = np.clip(values, 1e-6, None) ** strength
    elif kind == "bias":
        grid = np.exp(strength * rng.randn(3, 3, 3)).astype(np.float32)
        field = ndimage.zoom(grid, [s / 3 for s in image.shape], order=1)
        field = np.pad(field, [(0, s - f) for s, f in zip(image.shape, field.shape)], mode="edge")[:image.shape[0], :image.shape[1], :image.shape[2]]
        out[brain] = values * field[brain]
    elif kind == "contrast":
        mean = values.mean()
        out[brain] = (values - mean) * strength + mean
    elif kind == "noise":
        out[brain] = values + rng.randn(len(values)).astype(np.float32) * strength * values.std()
    else:
        raise ValueError(kind)
    out[brain] = np.clip(out[brain], 1e-6, 1.0)
    out[~brain] = 0
    return out


def thick_slices(k: int, volume: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Average over k consecutive slices in z and interpolate back to the original slices (same as the training augmentation)."""
    import torch.nn.functional as F
    x = torch.from_numpy(volume)[None, None]
    pooled = F.avg_pool3d(x, (k, 1, 1), stride=(k, 1, 1), ceil_mode=True, count_include_pad=False)
    back = F.interpolate(pooled, size=volume.shape, mode="trilinear", align_corners=False)[0, 0].numpy()
    return np.where(brain, back, 0).astype(np.float32)


def rotate(volume: np.ndarray, angle: float, order: int) -> np.ndarray:
    return ndimage.rotate(volume, angle, axes=IN_PLANE, reshape=False, order=order, mode="constant", cval=0.0)


def zoom_in_plane(volume: np.ndarray, factor: float, order: int) -> np.ndarray:
    """Zoom the in-plane axes about the centre, keeping the array shape (crop or zero-pad)."""
    zoomed = ndimage.zoom(volume, (1, factor, factor), order=order, mode="constant", cval=0.0)
    out = np.zeros_like(volume)
    src, dst = [], []
    for s_out, s_in in zip(volume.shape, zoomed.shape):
        n = min(s_out, s_in)
        a, b = (s_out - n) // 2, (s_in - n) // 2
        dst.append(slice(a, a + n)); src.append(slice(b, b + n))
    out[tuple(dst)] = zoomed[tuple(src)]
    return out


# --------------------------------------------------------------------------- scoring
def post_process(probability: np.ndarray, seg) -> np.ndarray:
    labels, n = ndimage.label(probability > THRESHOLD, structure=CONNECTIVITY_26)
    size = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (size * VOXEL_MM3 >= MIN_MM3) & (size <= MAX_VOXELS); keep[0] = False
    if seg is not None:
        for c, box in enumerate(ndimage.find_objects(labels), start=1):
            if keep[c] and majority_label(labels[box] == c, seg[box]) in CSF_LABELS:
                keep[c] = False
    return keep[labels]


def counts(reference: np.ndarray, prediction: np.ndarray) -> Tuple[int, int, int]:
    from metric import hits
    h = hits(reference, prediction)
    return int(h["tp"]), int(h["fp"]), int(h["fn"])


def pooled(rows: List[Tuple[int, int, int]]) -> Dict[str, float]:
    tp, fp, fn = (sum(r[i] for r in rows) for i in range(3))
    return dict(f1=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1), sensitivity=tp / max(tp + fn, 1),
                fp_per_case=fp / max(len(rows), 1), tp=tp, fp=fp, fn=fn)


def paired(a: Dict[str, Tuple[int, int, int]], b: Dict[str, Tuple[int, int, int]], draws: int = 5000) -> Tuple[float, float, float]:
    ids = sorted(a); A = [a[i] for i in ids]; B = [b[i] for i in ids]
    rng = np.random.RandomState(0)
    deltas = []
    for _ in range(draws):
        ix = rng.randint(len(ids), size=len(ids))
        deltas.append(pooled([A[j] for j in ix])["f1"] - pooled([B[j] for j in ix])["f1"])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return pooled(A)["f1"] - pooled(B)["f1"], float(lo), float(hi)


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="finished run: fold<k>/modell.pt and fold<k>/<id>_pred_proba.nii.gz")
    parser.add_argument("--net", required=True)
    parser.add_argument("--grid", required=True, help="VALDO grid folder, e.g. dev/gitter_d")
    parser.add_argument("--synthseg", help="path pattern with {id}; if given, the CSF rule is applied")
    parser.add_argument("--only", nargs="*", help="subset of perturbations: " + " ".join(TABLE))
    parser.add_argument("--cases", nargs="*", help="only these cases (smoke test)")
    parser.add_argument("--seed", type=int, default=0, help="random fields and noise are drawn once per case and strength from this seed")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    import cv5
    import turnier                                   # noqa: F401  (registers the tournament networks)
    from netze import NETZE

    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    run = absolute(args.run)
    cv5.GITTER = absolute(args.grid)
    cv5.lade_daten()
    records = cv5.FA["recs"]
    kinds = args.only or list(TABLE)
    unknown = [k for k in kinds if k not in TABLE]
    if unknown:
        raise SystemExit(f"unknown perturbation(s): {unknown}")
    conditions = [("none", 0.0)] + [(k, s) for k in kinds for s in TABLE[k]]

    per_case: Dict[str, Dict[str, Tuple[int, int, int]]] = {f"{k}:{s:g}": {} for k, s in conditions}
    done = 0
    for fold_dir in sorted(glob.glob(f"{run}/fold[0-9]")):
        case_ids = sorted(os.path.basename(p)[:-len("_pred_proba.nii.gz")] for p in glob.glob(f"{fold_dir}/*_pred_proba.nii.gz"))
        if args.cases:
            case_ids = [c for c in case_ids if c in args.cases]
        if not case_ids:
            continue
        net = NETZE[args.net](2)
        net.load_state_dict(torch.load(f"{fold_dir}/modell.pt", map_location="cpu"))
        net = net.to(cv5.DEV).eval()
        for case_id in case_ids:
            record = records[case_id]
            image0, frst0, reference = record["img"], record["frs"], record["lab"] > 0
            brain = image0 != 0
            seg = None
            if args.synthseg:
                import nibabel as nib
                like = nib.load(f"{fold_dir}/{case_id}_pred_proba.nii.gz")
                seg = np.ascontiguousarray(load_synthseg(synthseg_path(args.synthseg, case_id), like).transpose(2, 1, 0))   # -> (z, y, x)
            for kind, strength in conditions:
                rng = np.random.RandomState(args.seed * 100003 + hash((case_id, kind, float(strength))) % 100003)
                image, frst = image0, frst0
                if kind in ("gamma", "bias", "contrast", "noise"):
                    image = photometric(kind, strength, image0, rng)
                elif kind == "thick":
                    image, frst = thick_slices(int(strength), image0, brain), thick_slices(int(strength), frst0, brain)
                elif kind == "rotate":
                    image, frst = rotate(image0, strength, 1) * rotate(brain.astype(np.float32), strength, 0), rotate(frst0, strength, 1)
                elif kind == "scale":
                    image, frst = zoom_in_plane(image0, strength, 1), zoom_in_plane(frst0, strength, 1)
                record["img"], record["frs"] = np.ascontiguousarray(image, np.float32), np.ascontiguousarray(frst, np.float32)
                try:
                    probability, _ = cv5.proba_fall(net, case_id, nch=2)
                finally:
                    record["img"], record["frs"] = image0, frst0
                if kind == "rotate":
                    probability = rotate(probability, -strength, 1)
                elif kind == "scale":
                    probability = zoom_in_plane(probability, 1 / strength, 1)
                per_case[f"{kind}:{strength:g}"][case_id] = counts(reference, post_process(probability, seg))
            done += 1
            print(f"[{done}] {os.path.basename(fold_dir)} {case_id}: " + "  ".join(f"{k}:{s:g}={per_case[f'{k}:{s:g}'][case_id]}" for k, s in conditions[:3]) + " ...", flush=True)

    out = dict(run=run, net=args.net, csf_rule=bool(args.synthseg), cases=done, conditions={})
    base = per_case["none:0"]
    print(f"\n{'condition':14s} {'F1':>6s} {'P':>5s} {'S':>5s} {'FP/case':>8s}   delta vs none [95 %]")
    for key, rows in per_case.items():
        score = pooled(list(rows.values()))
        delta, lo, hi = paired(rows, base) if key != "none:0" else (0.0, 0.0, 0.0)
        out["conditions"][key] = dict(score, delta_f1=round(delta, 3), ci95=[round(lo, 3), round(hi, 3)])
        print(f"{key:14s} {score['f1']:6.3f} {score['precision']:5.2f} {score['sensitivity']:5.2f} {score['fp_per_case']:8.2f}   {delta:+.3f} [{lo:+.3f}; {hi:+.3f}]")
    json.dump(out, open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
