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

ROOT = "/home/uchralt/data/work/valdo-t2s"
CONNECTIVITY_26 = np.ones((3, 3, 3), bool)
CSF_LABELS = [4, 5, 14, 15, 24, 43, 44]
MAX_COMPONENT_VOXELS = 2100              # larger than a 10 mm sphere: not a microbleed
HALF = (20, 20, 10)                      # stored patch: 40 x 40 x 20 voxels = 20 x 20 x 20 mm (x, y, z);
                                         # training crops 32 x 32 x 16 from it at a random offset.
                                         # --half 32 32 16 stores 32 mm cubes for the two-scale CNN (cmb.stage2.classifier)


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
            majority = -1                                   # -1 = no tissue map, 0 = background only
            if seg is not None:
                tissue = seg[box][mask]; tissue = tissue[tissue > 0]
                majority = int(np.bincount(tissue).argmax()) if len(tissue) else 0
                if majority in CSF_LABELS and not args.keep_csf:
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
