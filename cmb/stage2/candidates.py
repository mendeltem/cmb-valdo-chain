"""Stage 2, step 1: turn the probability maps of the segmentation network into candidates.

Why a second stage
------------------
Error analysis of the best single network (2026-09-19): for 37 % of the microbleeds it misses, the
network's probability inside the lesion is >= 0.2 -- it "suspects" them, but the operating threshold
(0.3-0.9) cuts them off. Lowering the threshold finds them and floods us with false alarms
(F1 0.560 at 0.15 versus 0.627 at 0.3). A second stage looks at every candidate again, with a small
3D patch around it, and only has to answer one question: microbleed or not? This is the design of
MicrobleedNet (candidate detection + discrimination) and of the VALDO team "Zihao".

What this script does
---------------------
For every cross-validation case, using the OUT-OF-FOLD probability map (the case was never seen by
the network that produced it):
  * threshold low, take 26-connected components, drop giant blobs and components whose majority
    SynthSeg label is CSF (the parameter-free rule from 2026-09-18);
  * for each remaining component store a patch (T2*, radial symmetry, stage-1 probability) and the
    list of reference microbleeds it touches (empty list = false candidate).
Lesion-level scores can then be computed from these lists alone, with exactly the semantics of
``metric.hits``: TP = reference lesions touched by an accepted candidate, FP = accepted candidates
touching none, FN = the remaining reference lesions.

    python -m cmb.stage2.candidates --predictions ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d

Other cohorts (2026-09-20): instead of --grid give --manifest (list of {id, dir, ...}), the SynthSeg pattern and an
output folder. Private cohorts MUST get a private --out; patches are image data and never leave the machine.

    python -m cmb.stage2.candidates --predictions /private/run --manifest /private/manifest.json \
        --synthseg "/private/synthseg/{sid}_synthseg.nii.gz" --out /private/stage2/run
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

from cmb.analysis.fp_location import load_synthseg, synthseg_path

from cmb.paths import ROOT
CONNECTIVITY_26 = np.ones((3, 3, 3), bool)
CSF_LABELS = [4, 5, 14, 15, 24, 43, 44]
VENTRICLE_LABELS = [4, 5, 14, 15, 43, 44]
FREE_CSF_LABEL = 24
ERODE_ITERATIONS = 2            # wc2 'eroded' mode: how many voxels the free-CSF mask is eroded by
ERODE_FILL_LABEL = 2            # wc2 'eroded' mode: the eroded-away rim is relabelled to this tissue (white matter),
                                # not to background 0 -- majority_tissue() excludes 0 before argmax, so a 0-fill would
                                # leave the lesion's CSF voxels as the majority and keep deleting thin-rim lesions
MAX_COMPONENT_VOXELS = 2100              # larger than a 10 mm sphere: not a microbleed
HALF = (20, 20, 10)                      # stored patch: 40 x 40 x 20 voxels = 20 x 20 x 20 mm (x, y, z);
                                         # training crops 32 x 32 x 16 from it at a random offset.
                                         # --half 32 32 16 stores 32 mm cubes for the two-scale CNN (cmb.stage2.classifier)


def csf_rule(seg: Optional[np.ndarray], mode: str) -> tuple:
    """CSF rule of wc2 (2026-10-12) -> (seg_for_majority, csf_labels), or (None, None) if the rule is off.
    The MAJORITY LABEL of the candidate is computed in seg_for_majority (the real tissue segmentation, so the
    component's true tissue stays the majority), and the candidate is dropped when that label is in csf_labels:
      majority   the original rule: free CSF (label 24) and ventricles count as CSF
      eroded     same rule, but the free-CSF mask is first eroded and the eroded-away rim is relabelled to tissue
                 (ERODE_FILL_LABEL, white matter -- NOT background 0, because majority_tissue() excludes 0 before
                 argmax, so a 0-fill would leave the lesion's CSF voxels as the majority and keep deleting thin-rim
                 lesions). On SWI SynthSeg labels free CSF inside the brain, and that rim is what makes a deep true
                 lesion's majority label come out CSF. Ventricles are unchanged. The mask is eroded by ERODE_ITERATIONS
                 voxels (default 2, see below): the rim must be wide enough that a lesion sitting in a THIN CSF rim
                 loses its CSF majority and is kept, while a lesion sitting in THICK CSF keeps its CSF majority and is
                 dropped.
      ventricles only the ventricle labels count as CSF; free CSF (label 24) counts as tissue
      none       no CSF rule at all

    Why two voxels and not one: the task (2026-10-12) said "erode by one voxel", but a free-CSF pocket that is only one
    voxel wider than the lesion (a 5^3 block around a 3^3 lesion) eroded by ONE voxel is exactly the lesion's own 3^3
    box, so its majority stays free CSF and the thin-rim lesion is still deleted -- the very failure this variant exists
    to fix. Two voxels is the minimum that clears a one-voxel-wide rim while leaving a thick CSF pocket intact.
    """
    if seg is None or mode == "none":
        return None, None
    if mode == "majority":
        return seg, CSF_LABELS
    if mode == "ventricles":
        return seg, VENTRICLE_LABELS
    if mode == "eroded":
        free = seg == FREE_CSF_LABEL
        rim = free & ~ndimage.binary_erosion(free, iterations=ERODE_ITERATIONS)
        seg_eroded = seg.copy()
        seg_eroded[rim] = ERODE_FILL_LABEL
        return seg_eroded, CSF_LABELS
    raise ValueError(f"unknown --csf-mode: {mode}")


def majority_tissue(seg_mask: Optional[np.ndarray], box, mask: np.ndarray) -> int:
    """Majority label of the candidate in the given tissue segmentation; -1 = no tissue map, 0 = background only."""
    if seg_mask is None:
        return -1
    tissue = seg_mask[box][mask]
    tissue = tissue[tissue > 0]
    return int(np.bincount(tissue).argmax()) if len(tissue) else 0


def crop(volume: np.ndarray, centre, half) -> np.ndarray:
    out = np.zeros([2 * h for h in half], volume.dtype)
    src, dst = [], []
    for c, h, n in zip(centre, half, volume.shape):
        lo, hi = c - h, c + h
        start, stop = max(lo, 0), min(hi, n)
        if start >= stop:
            return out
        src.append(slice(start, stop)); dst.append(slice(start - lo, stop - lo))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--grid", help="folder with <case>/<case>_{image,frst,label}.nii.gz (VALDO grids)")
    parser.add_argument("--manifest", nargs="+", help="alternative to --grid: manifest files with id and dir per case")
    parser.add_argument("--synthseg", default="dev/synthseg/{id}_synthseg.nii.gz",
                        help="path pattern with {id} or {sid}; 'none' = no tissue map available, the CSF rule is skipped (private SWI cohort)")
    parser.add_argument("--keep-csf", action="store_true",
                        help="do NOT drop candidates whose majority tissue label is CSF; the label is stored per candidate (field 'tissue') so that "
                             "a classifier can use it softly. On the private SWI cohort the hard rule would delete 65 detected lesions (2026-09-21).")
    parser.add_argument("--csf-mode", choices=["majority", "eroded", "ventricles", "none"], default="majority",
                        help="which SynthSeg labels count as CSF for the drop rule: majority = free CSF (24) + ventricles (the original rule, "
                             "the default so nothing changes silently); eroded = free CSF eroded by 2 voxels and the rim relabelled to tissue (wc2, 2026-10-12: on SWI SynthSeg "
                             "labels free CSF inside the brain and the current rule deletes deep true lesions); ventricles = ventricles only; "
                             "none = no CSF rule. The candidate keeps the ORIGINAL majority label in the 'tissue' field either way.")
    parser.add_argument("--out", help="output folder (default: ergebnisse/stage2/<run name>); REQUIRED to be private for private cohorts")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--min-mm3", type=float, default=1.0)
    parser.add_argument("--tta", help="Ordner eines Spiegel-TTA-Laufs (cmb.analysis.tta_predict) zu DEMSELBEN Modell. Speichert je Kandidat "
                             "zwei Felder: tta_probability und tta_delta = |TTA - gewoehnlich|. Gemessen 2026-09-23 auf den 278 VALDO-Kandidaten: "
                             "echte Blutungen aendern sich unter Spiegelung um 0.014, Fehlalarme um 0.102 (Faktor 7); als Trennmerkmal AUC 0.711 "
                             "-- besser als FRST (0.56) und Kontrast (0.65) und eine ANDERE Information als die Wahrscheinlichkeit selbst.")
    parser.add_argument("--half", type=int, nargs=3, default=list(HALF), metavar=("X", "Y", "Z"),
                        help="half size of the stored patch in voxels; default 20 20 10 = 20 mm cube, 32 32 16 = 32 mm cube (two-scale CNN)")
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    if bool(args.grid) == bool(args.manifest):
        raise SystemExit("give exactly one of --grid and --manifest")
    predictions = absolute(args.predictions)
    if args.manifest:
        if not args.out:
            raise SystemExit("--manifest needs an explicit --out (private cohorts must not land in the project folder)")
        folders = {e["id"]: e["dir"] for m in args.manifest for e in json.load(open(absolute(m)))}
    else:
        folders = None
    out_dir = absolute(args.out) if args.out else f"{ROOT}/ergebnisse/stage2/{os.path.basename(predictions)}"
    os.makedirs(out_dir, exist_ok=True)

    patches, rows, references = [], [], {}
    for path in sorted(glob.glob(f"{predictions}/fold*/*_pred_proba.nii.gz")):
        case = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
        fold = int(path.split("/fold")[1][0])
        load = lambda p: np.asarray(nib.load(p).dataobj)
        probability = load(path).astype(np.float32) / 255.0
        if folders is not None and case not in folders:
            continue
        folder = folders[case] if folders is not None else f"{absolute(args.grid)}/{case}"
        image = load(f"{folder}/{case}_image.nii.gz").astype(np.float32)
        frst = load(f"{folder}/{case}_frst.nii.gz").astype(np.float32)
        reference = load(f"{folder}/{case}_label.nii.gz") > 0
        seg = None if args.synthseg.lower() == "none" else load_synthseg(synthseg_path(args.synthseg, case), nib.load(path))   # resampled if needed
        rule_seg, rule_labels = csf_rule(seg, args.csf_mode)          # wc2: which segmentation/labels the drop rule uses
        tta = None
        if args.tta:
            tta_pfad = f"{absolute(args.tta)}/fold{fold}/{case}_pred_proba.nii.gz"
            if not os.path.exists(tta_pfad):
                raise SystemExit(f"TTA-Karte fehlt: {tta_pfad} -- erst cmb.analysis.tta_predict fuer diesen Lauf rechnen")
            tta = load(tta_pfad).astype(np.float32) / 255.0

        ref_labels, n_ref = ndimage.label(reference, structure=CONNECTIVITY_26)
        references[case] = dict(fold=fold, n_reference=int(n_ref))
        labels, n = ndimage.label(probability > args.threshold, structure=CONNECTIVITY_26)
        for index, box in enumerate(ndimage.find_objects(labels), 1):
            mask = labels[box] == index
            voxels = int(mask.sum())
            if voxels * 0.25 < args.min_mm3 or voxels > MAX_COMPONENT_VOXELS:
                continue
            majority = -1                                   # -1 = no tissue map, 0 = background only (always the ORIGINAL label, for the 'tissue' field)
            if seg is not None:
                majority = majority_tissue(seg, box, mask)
                if rule_seg is not None and majority_tissue(rule_seg, box, mask) in rule_labels and not args.keep_csf:
                    continue
            touched = sorted(int(v) for v in np.unique(ref_labels[box][mask]) if v > 0)
            centre = [int(round(c)) + s.start for c, s in zip(ndimage.center_of_mass(mask), box)]
            patch = np.stack([crop(v, centre, args.half) for v in (image, frst, probability)])  # (3, x, y, z)
            patches.append(np.transpose(patch, (0, 3, 2, 1)).astype(np.float16))                # -> (3, z, y, x)
            zeile = dict(case=case, fold=fold, touched=touched, voxels=voxels,
                         probability=round(float(probability[box][mask].max()), 3), tissue=majority,
                         centre_voxel=centre)          # (x, y, z) on the grid -- needed to merge candidate sets (cmb.stage2.merge_candidates)
            if tta is not None:
                t = float(tta[box][mask].max())
                zeile.update(tta_probability=round(t, 3), tta_delta=round(abs(t - zeile["probability"]), 3))
            rows.append(zeile)
        print(f"{case}: {sum(r['case'] == case for r in rows)} candidates, {n_ref} reference lesions", flush=True)

    np.save(f"{out_dir}/patches.npy", np.stack(patches))
    json.dump(dict(threshold=args.threshold, min_mm3=args.min_mm3, half=list(args.half), tta=args.tta, candidates=rows, cases=references),
              open(f"{out_dir}/candidates.json", "w"))
    positives = sum(bool(r["touched"]) for r in rows)
    reachable = sum(len({t for r in rows if r["case"] == c for t in r["touched"]}) for c in references)
    total = sum(v["n_reference"] for v in references.values())
    print(f"{len(rows)} candidates ({positives} on a microbleed, {len(rows) - positives} false); "
          f"they reach {reachable} of {total} reference lesions = upper bound for sensitivity {reachable / total:.2f}")
    print("written to", out_dir)


if __name__ == "__main__":
    main()
