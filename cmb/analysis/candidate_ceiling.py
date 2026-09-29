"""How many reference microbleeds can a two-stage chain reach AT ALL -- as a function of the candidate threshold, and
pooled over several first stages.

Why (2026-09-22): the second stage can only reject false candidates, never invent a missed lesion. So the ceiling of the
whole chain is set by the candidate step: with the base run at threshold 0.15, 103 of 139 lesions are touched by some
candidate (F1 ceiling 0.851 even with a perfect classifier -- we are at 0.647). Two cheap questions follow, and neither had
been measured:

  1. Is the ceiling raised by a LOWER candidate threshold, and at what price in candidates?
  2. Do different training seeds miss the SAME lesions? If not, the UNION of their candidates has a higher ceiling than any
     single run -- unlike the probability-map ensemble (cmb.analysis.ensemble), which averages and therefore suppresses what
     only one member sees.

Both are answered from the stored probability maps alone: no training, no patches, no GPU.

    python -m cmb.analysis.candidate_ceiling --run s42=ergebnisse/turnier/a03-aniso-e60-gd s1=... s2=... \
        --grid dev/gitter_d --thresholds 0.15 0.10 0.05 0.02 --json ergebnisse/analysis/candidate_ceiling_valdo.json

Reported per run and threshold: candidates, reached reference lesions, ceiling sensitivity, and the F1 the chain would reach
with a PERFECT second stage (every false candidate rejected, every true one kept). For the union: the same, plus how many
lesions each run contributes that no other run reaches.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Set

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, ROOT)
from cmb.transfer.evaluate import CONNECTIVITY_26, MAX_VOXELS, MIN_MM3, VOXEL_MM3    # noqa: E402


def reached(probability: np.ndarray, reference_labels: np.ndarray, threshold: float, min_mm3: float) -> tuple:
    """-> (set of reference labels touched by a surviving candidate, number of candidates)."""
    labels, n = ndimage.label(probability > threshold, structure=CONNECTIVITY_26)
    size = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (size * VOXEL_MM3 >= min_mm3) & (size <= MAX_VOXELS); keep[0] = False
    surviving = keep[labels]
    touched = np.unique(reference_labels[surviving & (reference_labels > 0)])
    return {int(t) for t in touched}, int(keep.sum())


def perfect_chain_f1(reached_count: int, total: int) -> float:
    """F1 of the chain if the second stage were perfect: TP = reached, FP = 0, FN = total - reached."""
    return 2 * reached_count / (2 * reached_count + (total - reached_count))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, help="NAME=FOLDER (folders with fold<k>/<id>_pred_proba.nii.gz)")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.10, 0.05, 0.02])
    parser.add_argument("--min-mm3", type=float, default=1.0, help="same as cmb.stage2.candidates")
    parser.add_argument("--json")
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    grid = absolute(args.grid)
    runs = dict(spec.split("=", 1) for spec in args.run)

    # reference lesions once, keyed by "<case>#<label>"
    reference: Dict[str, np.ndarray] = {}
    lesions: Set[str] = set()
    for case_dir in sorted(glob.glob(f"{grid}/*/")):
        case = os.path.basename(case_dir.rstrip("/"))
        path = f"{case_dir}{case}_label.nii.gz"
        if not os.path.exists(path):
            continue
        labels, n = ndimage.label(np.asarray(nib.load(path).dataobj) > 0, structure=CONNECTIVITY_26)
        reference[case] = labels
        lesions.update(f"{case}#{i}" for i in range(1, n + 1))

    out: Dict = dict(grid=grid, thresholds=args.thresholds, runs={}, union={})
    per_threshold: Dict[float, Dict[str, Set[str]]] = {t: {} for t in args.thresholds}
    counts: Dict[float, Dict[str, int]] = {t: {} for t in args.thresholds}
    cases_seen: Set[str] = set()

    for name, folder in runs.items():
        hits = {t: set() for t in args.thresholds}
        candidates = {t: 0 for t in args.thresholds}
        for path in sorted(glob.glob(f"{absolute(folder)}/fold[0-9]/*_pred_proba.nii.gz")):
            case = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
            if case not in reference:
                continue
            cases_seen.add(case)
            probability = np.asarray(nib.load(path).dataobj).astype(np.float32) / 255.0
            for t in args.thresholds:
                touched, n_cand = reached(probability, reference[case], t, args.min_mm3)
                hits[t].update(f"{case}#{i}" for i in touched)
                candidates[t] += n_cand
        for t in args.thresholds:
            per_threshold[t][name] = hits[t]
            counts[t][name] = candidates[t]
        print(f"{name}: " + "  ".join(f"t={t:g}: {candidates[t]} cand., {len(hits[t])} reached" for t in args.thresholds), flush=True)

    total = len({l for l in lesions if l.split("#")[0] in cases_seen})
    print(f"\n{len(cases_seen)} cases, {total} reference microbleeds\n")
    print(f"{'first stage':22s} {'threshold':>9s} {'candidates':>11s} {'reached':>9s} {'ceil. S':>8s} {'F1 ceil.':>9s}")
    for name in runs:
        for t in args.thresholds:
            k = len(per_threshold[t][name])
            out["runs"].setdefault(name, {})[f"{t:g}"] = dict(candidates=counts[t][name], reached=k,
                                                              ceiling_sensitivity=round(k / total, 3), perfect_f1=round(perfect_chain_f1(k, total), 3))
            print(f"{name:22s} {t:9g} {counts[t][name]:11d} {k:9d} {k / total:8.3f} {perfect_chain_f1(k, total):9.3f}")

    if len(runs) > 1:
        print()
        for t in args.thresholds:
            union = set().union(*per_threshold[t].values())
            cand = sum(counts[t].values())
            only = {name: len(h - set().union(*(o for n, o in per_threshold[t].items() if n != name))) for name, h in per_threshold[t].items()}
            best = max(len(h) for h in per_threshold[t].values())
            out["union"][f"{t:g}"] = dict(candidates_summed=cand, reached=len(union), ceiling_sensitivity=round(len(union) / total, 3),
                                          perfect_f1=round(perfect_chain_f1(len(union), total), 3), gain_over_best=len(union) - best, only_this_run=only)
            print(f"{'VEREINIGUNG':22s} {t:9g} {cand:11d} {len(union):9d} {len(union) / total:8.3f} {perfect_chain_f1(len(union), total):9.3f}"
                  f"   (+{len(union) - best} gegen den besten Einzellauf; nur von einem Lauf erreicht: {only})")
    if args.json:
        out["total_lesions"] = total; out["cases"] = len(cases_seen)
        json.dump(out, open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
