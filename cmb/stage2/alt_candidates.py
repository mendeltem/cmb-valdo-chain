"""A second, independent candidate source united with the U-Net map -- to lift the ceiling of the two-stage chain.

Why (2026-10-25, wiki cmb_detection/search-space-expansion-and-novel-hybrids-2026-10, Phase 0): the second stage can only
reject candidates, so the chain is capped by the lesions the candidates reach (VALDO: 103 of 139 at threshold 0.15, F1
ceiling 0.85). The union of SEEDS was measured (2026-09-22/28): different seeds of the same U-Net miss similar lesions, the
ceiling rose but the chain did not. A source that does not come from a U-Net at all may miss DIFFERENT lesions:

  frst   peaks of the radial symmetry map already stored on the grid: voxels inside the brain whose FRST value is at least
         --frst-fraction x the --frst-quantile percentile of the brain (default: the top 0.1 %). The FRST/fast radial symmetry
         candidate step is the classic CMB generator (Liu 2019, Xia 2024), never used here as an independent source.
  gmm    per-scan adaptive threshold after Bogdanov 2026 (bioRxiv 10.64898/2026.07.24.740624): three-component Gaussian mixture
         of the brain intensities, the middle component taken as grey matter; a voxel is a seed if the (inverted, so dark on
         T2* = bright here) image exceeds mu + --gmm-k sigma of that component AND its in-plane local contrast (image minus a
         9 x 9 in-plane mean) exceeds --gmm-k times the robust spread (1.4826 MAD) of the local contrast in the brain.
  both   union of the two.

Output: a run folder in the usual layout (fold<k>/<id>_pred_proba.nii.gz, 8 bit) with  p = max(p_unet, VALUE x alt)  where
VALUE = --value (default 0.16, just above the candidate threshold 0.15). So every U-Net candidate stays exactly as it was (its
probability channel is the U-Net's wherever the U-Net is higher), and the new components enter with a low, honest stage-1
probability -- the second stage sees that the U-Net did not flag them. Every downstream tool works unchanged:
cmb.analysis.candidate_ceiling (VALDO grid), cmb.stage2.candidates (+ upper bound printout), cmb.stage2.pooled.

    python -m cmb.stage2.alt_candidates --base ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d --mode frst \
        --out ergebnisse/turnier/alt-frst-a03-aniso-e60-gd
    python -m cmb.stage2.alt_candidates --base /private/transfer/run --manifest /private/manifest.json --mode gmm --out /private/alt-gmm-run

--max-new K (added 2026-10-25 after wc3): keep only the K strongest NEW components per scan (ranked by their FRST peak), each at least
--min-voxels large. Uncapped, the sources gave 33,661-125,920 candidates on VALDO instead of 278 (ceiling F1 0.851 -> 0.92-0.95)
and building the candidate patches for the private cohorts ran the machine out of memory three times. K = 0 keeps everything
(the wc3 behaviour); for a stage-2 run use a K that keeps the candidate count within about twice the U-Net's.

CPU only. Private cohorts MUST get a private --out (the maps are image-derived). Prints totals only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

from cmb.paths import ROOT


def frst_seeds(frst: np.ndarray, brain: np.ndarray, quantile: float, fraction: float) -> np.ndarray:
    if not brain.any():
        return np.zeros_like(brain)
    level = np.percentile(frst[brain], quantile) * fraction
    return brain & (frst >= level) & (frst > 0)


def gmm_seeds(image: np.ndarray, brain: np.ndarray, k: float, seed: int = 0) -> np.ndarray:
    from sklearn.mixture import GaussianMixture
    values = image[brain].astype(np.float64)
    if len(values) < 1000:
        return np.zeros_like(brain)
    rng = np.random.RandomState(seed)
    sample = values[rng.choice(len(values), min(len(values), 200_000), replace=False)][:, None]
    gmm = GaussianMixture(3, random_state=seed).fit(sample)
    order = np.argsort(gmm.means_[:, 0])
    mu = float(gmm.means_[order[1], 0]); sigma = float(np.sqrt(gmm.covariances_[order[1]].ravel()[0]))
    local = image - ndimage.uniform_filter(image, size=(1, 9, 9))
    inside = local[brain]
    noise = 1.4826 * float(np.median(np.abs(inside - np.median(inside))))      # robust spread of the local contrast
    return brain & (image > mu + k * sigma) & (local > k * noise)


def cap_components(new: np.ndarray, strength: np.ndarray, max_new: int, min_voxels: int) -> np.ndarray:
    """Components of `new` (26-connected) with >= min_voxels voxels; if max_new > 0 only the max_new with the highest peak of
    `strength` inside them."""
    labels, n = ndimage.label(new, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return new
    idx = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(strength), labels, idx)
    peak = ndimage.maximum(strength, labels, idx)
    ok = idx[size >= min_voxels]
    if max_new > 0 and len(ok) > max_new:
        ok = ok[np.argsort(-np.asarray(peak)[ok - 1], kind="stable")[:max_new]]
    return np.isin(labels, ok)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="U-Net run folder with fold<k>/<id>_pred_proba.nii.gz (out-of-fold maps)")
    parser.add_argument("--grid", help="folder with <case>/<case>_{image,frst}.nii.gz (VALDO grids)")
    parser.add_argument("--manifest", nargs="+", help="alternative to --grid: manifests with id and dir per case")
    parser.add_argument("--mode", choices=["frst", "gmm", "both"], required=True)
    parser.add_argument("--frst-quantile", type=float, default=99.9)
    parser.add_argument("--frst-fraction", type=float, default=1.0)
    parser.add_argument("--gmm-k", type=float, default=2.0)
    parser.add_argument("--value", type=float, default=0.16, help="probability given to the new components (just above 0.15)")
    parser.add_argument("--max-new", type=int, default=0, help="keep the K strongest new components per scan by FRST peak (0 = all)")
    parser.add_argument("--min-voxels", type=int, default=4, help="drop new components smaller than this (4 = 1 mm3 on the grid)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    if bool(args.grid) == bool(args.manifest):
        raise SystemExit("give exactly one of --grid and --manifest")
    folders = {e["id"]: e["dir"] for m in args.manifest for e in json.load(open(absolute(m)))} if args.manifest else None
    out = absolute(args.out)
    if os.path.abspath(out) == os.path.abspath(absolute(args.base)):
        raise SystemExit("--out must differ from --base")
    total_new = total_cases = 0
    for path in sorted(glob.glob(f"{absolute(args.base)}/fold*/*_pred_proba.nii.gz")):
        case = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
        if folders is not None and case not in folders:
            continue
        folder = folders[case] if folders is not None else f"{absolute(args.grid)}/{case}"
        ref = nib.load(path)
        p = np.asarray(ref.dataobj).astype(np.float32) / 255.0
        image = np.asarray(nib.load(f"{folder}/{case}_image.nii.gz").dataobj).astype(np.float32)
        brain = image > 0
        seeds = np.zeros_like(brain)
        if args.mode in ("frst", "both"):
            frst = np.asarray(nib.load(f"{folder}/{case}_frst.nii.gz").dataobj).astype(np.float32)
            seeds |= frst_seeds(frst, brain, args.frst_quantile, args.frst_fraction)
        if args.mode in ("gmm", "both"):
            seeds |= gmm_seeds(image, brain, args.gmm_k)
        new = seeds & (p <= args.value)
        kept = cap_components(new, frst if args.mode != "gmm" else np.asarray(nib.load(f"{folder}/{case}_frst.nii.gz").dataobj, np.float32),
                              args.max_new, args.min_voxels)
        combined = np.maximum(p, args.value * kept)
        total_new += int(ndimage.label(kept, structure=np.ones((3, 3, 3), bool))[1]); total_cases += 1
        target = f"{out}/{os.path.basename(os.path.dirname(path))}/{case}_pred_proba.nii.gz"
        os.makedirs(os.path.dirname(target), exist_ok=True)
        img = nib.Nifti1Image(np.round(combined * 255).astype(np.uint8), ref.affine, ref.header)
        img.set_data_dtype(np.uint8)
        nib.save(img, target)
    json.dump(dict(base=args.base, mode=args.mode, frst_quantile=args.frst_quantile, frst_fraction=args.frst_fraction,
                   gmm_k=args.gmm_k, value=args.value, max_new=args.max_new, min_voxels=args.min_voxels, cases=total_cases), open(f"{out}/alt_candidates.json", "w"), indent=1)
    print(f"{total_cases} cases, {total_new} new seed components outside the U-Net map ({total_new / max(total_cases, 1):.1f} per case); "
          f"written to {out}")


if __name__ == "__main__":
    main()
