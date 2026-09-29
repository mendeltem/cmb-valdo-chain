"""How much of our error is the reference annotation? -- three estimates that need no human reader.

Why (2026-09-22): the blinded review page (``cmb.review.build_review --blind``) answers this properly, but it needs a
clinical reader, and the project has none available. Until one is, three objective proxies bound the question. None of them
is a verdict; together they say how much room there is for the reference to be wrong.

1. AGREEMENT WITH AN INDEPENDENT METHOD. SHIVA-CMB was trained on 450 other scans and has never seen our pipeline. A false
   positive of ours that SHIVA independently marks is evidence for a microbleed the annotators missed -- two systems with
   different training data agreeing against the reference. Conversely, a reference lesion that NEITHER method finds is either
   physically invisible at this resolution or not a microbleed.
2. AGREEMENT ACROSS TRAINING SEEDS. A false positive that all seeds of the same recipe find independently is a stable image
   feature, not a fluke of one training run. Weaker evidence than 1 (same data, same recipe), but it separates "the network
   sees something" from "this run happened to fire".
3. RESEMBLANCE TO TRUE LESIONS. Size, stage-1 probability and anatomical region of the false positives against the same
   measures for the true positives. A false positive indistinguishable from a true lesion on every measured property is
   suspicious; an elongated thing in white matter is a vessel.

Output: counts per class and a RANGE for the lesion-level F1 -- the measured value, and the value if every false positive
confirmed by SHIVA were counted as a true lesion (an upper bound for what this proxy can justify, not an estimate).

    python -m cmb.analysis.label_noise_proxies --run ergebnisse/turnier/a03-aniso-e60-gd \
        --seeds ergebnisse/turnier/a03-aniso-s1-e60-gd ergebnisse/turnier/a03-aniso-s2-e60-gd \
        --shiva ergebnisse/baselines/shiva-cmb-valdo --grid dev/gitter_d \
        --synthseg "dev/synthseg/{id}_synthseg.nii.gz" --json ergebnisse/analysis/label_noise_proxies_valdo.json

CPU only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Dict, List

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
sys.path.insert(0, ROOT)
from cmb.analysis.fp_location import load_synthseg, synthseg_path        # noqa: E402
from cmb.review.build_review import REGION, final_chain                  # noqa: E402
from cmb.transfer.evaluate import CONNECTIVITY_26                        # noqa: E402

NEAR_MM = 3.0                       # a hit of another method counts if its mask comes this close
VOXEL_MM = np.array([0.5, 0.5, 1.0])


def near(box, mask_box: np.ndarray, other: np.ndarray) -> bool:
    """Does ``other`` touch the component (given as ``mask_box`` inside ``box``) or come within NEAR_MM of it?

    Works on the bounding box plus a margin, not on the whole volume -- dilating 14 million voxels per component and per
    compared method would dominate the runtime.
    """
    iterations = int(round(NEAR_MM / VOXEL_MM.max()))
    wide = tuple(slice(max(s.start - iterations, 0), min(s.stop + iterations, n))
                 for s, n in zip(box, other.shape))
    here = np.zeros([s.stop - s.start for s in wide], bool)
    here[tuple(slice(s.start - w.start, s.stop - w.start) for s, w in zip(box, wide))] = mask_box
    if (here & other[wide]).any():
        return True
    return bool((ndimage.binary_dilation(here, structure=CONNECTIVITY_26, iterations=iterations) & other[wide]).any())


def region_of(mask: np.ndarray, seg) -> str:
    if seg is None:
        return "unbekannt"
    labels = seg[mask]; labels = labels[labels > 0]
    return REGION.get(int(np.bincount(labels).argmax()), "ausserhalb") if len(labels) else "ausserhalb"


def f1_of(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="the reported run (its errors are the ones we ask about)")
    parser.add_argument("--seeds", nargs="*", default=[], help="further runs of the SAME recipe, other seed")
    parser.add_argument("--shiva", help="folder with <case>_pred.nii.gz of an independent method")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--synthseg", help="path pattern with {id}")
    parser.add_argument("--json")
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    run, grid = absolute(args.run), absolute(args.grid)
    seeds = [absolute(p) for p in args.seeds]

    falsch: List[Dict] = []          # false positives of the reported run
    treffer: List[Dict] = []         # its true positives (for the resemblance comparison)
    verpasst: List[Dict] = []        # reference lesions it misses
    tp_total = fp_total = fn_total = 0

    for path in sorted(glob.glob(f"{run}/fold[0-9]/*_pred_proba.nii.gz")):
        case = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
        image = nib.load(path)
        probability = np.asarray(image.dataobj).astype(np.float32) / 255.0
        reference = np.asarray(nib.load(f"{grid}/{case}/{case}_label.nii.gz").dataobj) > 0
        seg = load_synthseg(synthseg_path(args.synthseg, case), image) if args.synthseg else None
        prediction = final_chain(probability, seg)

        andere = []                                        # same fixed chain, other seeds
        for other in seeds:
            hit = glob.glob(f"{other}/fold[0-9]/{case}_pred_proba.nii.gz")
            if hit:
                p = np.asarray(nib.load(hit[0]).dataobj).astype(np.float32) / 255.0
                andere.append(final_chain(p, seg))
        shiva = None
        if args.shiva:
            sp = f"{absolute(args.shiva)}/{case}_pred.nii.gz"
            if os.path.exists(sp):
                shiva = np.asarray(nib.load(sp).dataobj) > 0

        pred_labels, n_pred = ndimage.label(prediction, structure=CONNECTIVITY_26)
        ref_labels, n_ref = ndimage.label(reference, structure=CONNECTIVITY_26)
        getroffen = set()
        for index, box in enumerate(ndimage.find_objects(pred_labels), 1):
            mask_box = pred_labels[box] == index
            beruehrt = [int(v) for v in np.unique(ref_labels[box][mask_box]) if v > 0]
            eintrag = dict(case=case, voxels=int(mask_box.sum()),
                           probability=round(float(probability[box][mask_box].max()), 3),
                           region=region_of(mask_box, seg[box] if seg is not None else None),
                           n_seeds=sum(near(box, mask_box, a) for a in andere),
                           shiva=bool(shiva is not None and near(box, mask_box, shiva)))
            if beruehrt:
                getroffen.update(beruehrt); treffer.append(eintrag)
            else:
                falsch.append(eintrag); fp_total += 1
        for index, box in enumerate(ndimage.find_objects(ref_labels), 1):
            if index in getroffen:
                continue
            mask_box = ref_labels[box] == index
            verpasst.append(dict(case=case, voxels=int(mask_box.sum()),
                                 region=region_of(mask_box, seg[box] if seg is not None else None),
                                 n_seeds=sum(near(box, mask_box, a) for a in andere),
                                 shiva=bool(shiva is not None and near(box, mask_box, shiva))))
        tp_total += len(getroffen); fn_total += n_ref - len(getroffen)

    n_seeds_max = len(seeds)
    print(f"Reported run at the fixed cell: {tp_total} hits, {fp_total} false alarms, {fn_total} missed "
          f"-> F1 {f1_of(tp_total, fp_total, fn_total):.3f}\n")

    shiva_fp = sum(r["shiva"] for r in falsch)
    shiva_tp = sum(r["shiva"] for r in treffer)
    shiva_fn = sum(r["shiva"] for r in verpasst)
    print("1) Independent method (SHIVA-CMB)")
    print(f"   of {len(treffer)} hits SHIVA confirms {shiva_tp} ({shiva_tp / max(len(treffer),1):.0%})  <- how often the two agree, when the reference agrees too")
    print(f"   of {fp_total} false alarms SHIVA finds {shiva_fp} ({shiva_fp / max(fp_total,1):.0%})  <- candidates for missed annotations")
    print(f"   of {fn_total} missed reference microbleeds SHIVA finds {shiva_fn} ({shiva_fn / max(fn_total,1):.0%})  <- WE missed those, not the reference")
    print(f"   found by neither: {fn_total - shiva_fn} reference microbleeds")
    if n_seeds_max:
        print(f"\n2) Agreement across {n_seeds_max + 1} seeds of the same recipe")
        for name, rows in (("hits", treffer), ("false alarms", falsch), ("missed", verpasst)):
            z = Counter(r["n_seeds"] for r in rows)
            print(f"   {name:12s} confirmed by further seeds: " + ", ".join(f"{k} of {n_seeds_max}: {z.get(k,0)}" for k in range(n_seeds_max + 1)))
        alle = sum(1 for r in falsch if r["n_seeds"] == n_seeds_max)
        beides = sum(1 for r in falsch if r["n_seeds"] == n_seeds_max and r["shiva"])
        print(f"   -> {alle} false alarms are found by ALL seeds, of which {beides} also by SHIVA")

    print("\n3) Do the false alarms look like real microbleeds?")
    for name, rows in (("hits", treffer), ("false alarms", falsch)):
        if not rows:
            continue
        v = np.array([r["voxels"] for r in rows]); p = np.array([r["probability"] for r in rows])
        orte = Counter(r["region"] for r in rows)
        print(f"   {name:12s} n={len(rows):3d}  size {np.median(v):4.0f} voxels [{np.percentile(v,25):.0f}; {np.percentile(v,75):.0f}]  "
              f"probability {np.median(p):.2f}  location " + ", ".join(f"{k} {n}" for k, n in orte.most_common(4)))

    korrigiert = f1_of(tp_total + shiva_fp, fp_total - shiva_fp, fn_total)
    print(f"\nRange of the F1: measured {f1_of(tp_total, fp_total, fn_total):.3f}; "
          f"if EVERY false alarm confirmed by SHIVA were a real microbleed: {korrigiert:.3f} "
          f"(upper bound of what this substitution justifies -- not an estimate)")
    if args.json:
        json.dump(dict(run=run, seeds=seeds, shiva=args.shiva,
                       measured=dict(tp=tp_total, fp=fp_total, fn=fn_total, f1=round(f1_of(tp_total, fp_total, fn_total), 3)),
                       shiva_confirms=dict(true_positives=shiva_tp, false_positives=shiva_fp, missed=shiva_fn),
                       f1_if_shiva_confirmed_are_real=round(korrigiert, 3),
                       false_positives=falsch, missed=verpasst, true_positives=treffer),
                  open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
