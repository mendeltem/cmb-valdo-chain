"""Why are reference lesions missed?  (follow-up to fp_location, 2026-09-20)

fp_location showed WHERE lesions are missed (private T2* cohort: infratentorial sensitivity 0.46-0.50 against
0.71-0.79 elsewhere). This script asks WHY. Every reference lesion gets, once:

  location, structure   SynthSeg majority label under the lesion (infratentorial: cerebellar cortex / white matter / brainstem)
  volume_mm3            voxels x 0.25 mm3
  contrast              relative darkening against an in-plane ring 0.5-2 mm around the lesion, in ORIGINAL intensity
                        (the grid stores 1 - I/max, lesions bright):  (ring - lesion) / ring
  outside_fraction      share of lesion voxels where the image is 0, i.e. cut away by the brain mask -> invisible to the net
  border_mm             distance of the lesion to the edge of the brain mask (capped at 15 mm)
and per run: detected (same fixed post-processing as cmb.transfer.evaluate) and the highest probability on the lesion.

Tables (sums only -- safe for the private cohort; case ids appear in no table, only --case-list writes them, to a PRIVATE file):
  1  by location: sensitivity per run next to size, contrast, mask and border figures
  1b by size tercile and contrast half: sensitivity per run
  2  infratentorial lesions by structure
  3  missed against detected infratentorial lesions, and how close the net was (highest probability on the missed ones)
  4  is it the PLACE or the LESION?  sensitivity in strata of equal size and contrast, lobar against infratentorial, plus the
     infratentorial sensitivity one would expect if lobar rates applied in every stratum (direct standardisation)
  5  is it a few CASES?  share of the infratentorial misses held by the three cases with the most misses
  6  what if the threshold were lower ONLY in the infratentorial region?  EXPLORATORY: a rule picked here is selected on the
     evaluated cases and needs cross-fold selection before it counts as a result.

    python -m cmb.analysis.missed_lesions --run NAME=FOLDER [NAME=FOLDER ...] --manifest M.json \
        --synthseg "/path/{sid}_synthseg.nii.gz" [--json out.json] [--workers 4] [--case-list private.json]
"""
import argparse
import glob
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

from cmb.analysis.fp_location import ENGLISH, ROOT, components, f1, load_synthseg, padded, synthseg_path
from cmb.transfer.evaluate import CONNECTIVITY_26, MAX_VOXELS, MIN_MM3, THRESHOLD, VOXEL_MM3

from gewebekarte import ORT            # noqa: E402  (path set by fp_location): SynthSeg label -> German location name

STRUCTURE = {7: "cerebellar white matter", 46: "cerebellar white matter", 8: "cerebellar cortex", 47: "cerebellar cortex",
             16: "brainstem"}
REGION_THRESHOLDS = (THRESHOLD, 0.2, 0.1, 0.05)        # the first one is the baseline
BORDER_CAP_MM = 15.0
IN_PLANE_5, IN_PLANE_3 = np.ones((5, 5, 1), bool), np.ones((3, 3, 1), bool)


def majority_label(mask: np.ndarray, seg: np.ndarray) -> int:
    """Same rule as gewebekarte.ort_der_komponente: majority label under the lesion; background only -> widen by 3 voxels."""
    for iterations in (0, 3):
        m = ndimage.binary_dilation(mask, structure=CONNECTIVITY_26, iterations=iterations) if iterations else mask
        values = seg[m]; values = values[values > 0]
        if len(values):
            return int(np.bincount(values).argmax())
    return 0


def contrast_of(image: np.ndarray, mask: np.ndarray, other_lesions: np.ndarray):
    ring = ndimage.binary_dilation(mask, structure=IN_PLANE_5, iterations=2) & ~ndimage.binary_dilation(mask, structure=IN_PLANE_3)
    ring &= (image != 0) & ~other_lesions
    inside = mask & (image != 0)
    if ring.sum() < 8 or not inside.any():
        return None
    original_ring, original_lesion = 1.0 - float(image[ring].mean()), 1.0 - float(image[inside].mean())
    return (original_ring - original_lesion) / original_ring if original_ring > 0 else None


def keep_components(candidate: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(candidate, structure=CONNECTIVITY_26)
    size = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (size * VOXEL_MM3 >= MIN_MM3) & (size <= MAX_VOXELS); keep[0] = False
    return keep[labels]


def score(kept: np.ndarray, ref_labels: np.ndarray) -> Tuple[set, int]:
    """-> ids of the detected reference lesions (touch rule), number of predicted components touching no lesion."""
    detected = set(np.unique(ref_labels[kept]).tolist()) - {0}
    pred_labels, n_pred = ndimage.label(kept, structure=CONNECTIVITY_26)
    hitting = set(np.unique(pred_labels[ref_labels > 0]).tolist()) - {0}
    return detected, n_pred - len(hitting)


def process_case(job) -> Dict:
    case_id, case_dir, seg_path, proba_paths = job
    image_nii = nib.load(f"{case_dir}/{case_id}_image.nii.gz")
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    zooms = tuple(float(z) for z in image_nii.header.get_zooms()[:3])
    reference = np.asarray(nib.load(f"{case_dir}/{case_id}_label.nii.gz").dataobj) > 0
    seg = load_synthseg(seg_path, image_nii) if seg_path else np.zeros(image.shape, np.int32)     # no tissue map: every location reads "outside"
    ref_labels, n_ref, ref_boxes = components(reference)
    region = ndimage.binary_dilation(np.isin(seg, list(STRUCTURE)), iterations=2)

    lesions = []
    wide_margin = [int(math.ceil(BORDER_CAP_MM / z)) + 1 for z in zooms]
    for c in range(1, n_ref + 1):
        box = padded(ref_boxes[c - 1], ref_labels.shape, margin=6)
        mask = ref_labels[box] == c
        label = majority_label(mask, seg[box])
        wide = tuple(slice(max(s.start - m, 0), min(s.stop + m, dim)) for s, m, dim in zip(ref_boxes[c - 1], wide_margin, image.shape))
        brain = image[wide] != 0
        if brain.all():
            border = BORDER_CAP_MM
        else:
            border = min(float(ndimage.distance_transform_edt(brain, sampling=zooms)[ref_labels[wide] == c].min()), BORDER_CAP_MM)
        lesions.append(dict(location=ENGLISH[ORT.get(label, "ausserhalb")], structure=STRUCTURE.get(label),
                            volume_mm3=float(mask.sum() * VOXEL_MM3), contrast=contrast_of(image[box], mask, reference[box] & ~mask),
                            outside_fraction=float((image[box][mask] == 0).mean()), border_mm=border, runs={}))

    what_if = {}
    for name, path in proba_paths.items():
        probability = np.asarray(nib.load(path).dataobj).astype(np.float32) / 255.0
        above = probability > THRESHOLD
        what_if[name] = {}
        for t in REGION_THRESHOLDS:
            candidate = above if t == THRESHOLD else above | (region & (probability > t))
            detected, fp = score(keep_components(candidate), ref_labels)
            what_if[name][t] = dict(fp=fp, detected=[c in detected for c in range(1, n_ref + 1)])
        for c, lesion in enumerate(lesions, start=1):
            box = padded(ref_boxes[c - 1], ref_labels.shape, margin=6)
            lesion["runs"][name] = dict(detected=what_if[name][THRESHOLD]["detected"][c - 1],
                                        p_max=float(probability[box][ref_labels[box] == c].max()))
    return dict(case=case_id, lesions=lesions, what_if=what_if)


def median(values: List) -> float:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else float("nan")


def sensitivity(rows: List[Dict], run: str) -> float:
    return sum(r["runs"][run]["detected"] for r in rows) / len(rows) if rows else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, help="NAME=FOLDER")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--synthseg", help="path pattern with {id} or {sid}; without it only the tables that need no anatomy are meaningful (1b: size and contrast)")
    parser.add_argument("--json", help="write the tables here as well")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case-list", help="PRIVATE: per-case infratentorial counts WITH case ids, for looking the cases up in the "
                                            "QC page. Never point this at a folder that is published.")
    args = parser.parse_args()

    cases = {e["id"]: e for p in args.manifest for e in json.load(open(p))}
    runs: Dict[str, Dict[str, str]] = {}
    for spec in args.run:
        name, folder = spec.split("=", 1)
        folder = folder if os.path.isabs(folder) else f"{ROOT}/{folder}"
        runs[name] = {os.path.basename(p)[:-len("_pred_proba.nii.gz")]: p for p in glob.glob(f"{folder}/fold*/*_pred_proba.nii.gz")}
    names = list(runs)
    jobs, skipped = [], 0
    for case_id, case in sorted(cases.items()):
        seg_path = synthseg_path(args.synthseg, case_id) if args.synthseg else None
        if (seg_path and not os.path.exists(seg_path)) or any(case_id not in runs[n] for n in names):
            skipped += 1
            continue
        jobs.append((case_id, case["dir"], seg_path, {n: runs[n][case_id] for n in names}))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process_case, jobs))

    rows = [dict(lesion, case=r["case"]) for r in results for lesion in r["lesions"]]
    print(f"{len(results)} cases, {len(rows)} reference lesions" + (f", {skipped} cases skipped" if skipped else ""))
    out: Dict = dict(cases=len(results), lesions=len(rows), runs=names)

    # ---- 1  by location
    print("\n1  by location")
    print(f"{'location':16s} {'n':>4s}  " + "  ".join(f"sens {n[:12]:12s}" for n in names)
          + f"  {'med mm3':>8s} {'med contrast':>12s} {'>half outside mask':>18s} {'med border mm':>13s}")
    out["by_location"] = {}
    for location in ENGLISH.values():
        here = [r for r in rows if r["location"] == location]
        if not here:
            continue
        entry = dict(n=len(here), sensitivity={n: round(sensitivity(here, n), 3) for n in names},
                     median_volume_mm3=round(median([r["volume_mm3"] for r in here]), 2),
                     median_contrast=round(median([r["contrast"] for r in here]), 3),
                     more_than_half_outside_mask=sum(r["outside_fraction"] > 0.5 for r in here),
                     median_border_mm=round(median([r["border_mm"] for r in here]), 1))
        out["by_location"][location] = entry
        print(f"{location:16s} {len(here):4d}  " + "  ".join(f"{entry['sensitivity'][n]:17.2f}" for n in names)
              + f"  {entry['median_volume_mm3']:8.2f} {entry['median_contrast']:12.3f} {entry['more_than_half_outside_mask']:18d} "
                f"{entry['median_border_mm']:13.1f}")

    # ---- 1b  by size and by contrast, every run (does a change of the training signal reach the small, faint lesions?)
    print("\n1b  by size tercile and by contrast half, all locations")
    with_contrast = [r for r in rows if r["contrast"] is not None]
    cuts = np.percentile([r["volume_mm3"] for r in rows], [33.3, 66.7])
    contrast_median = float(np.median([r["contrast"] for r in with_contrast]))
    groups = [(f"small  (< {cuts[0]:.1f} mm3)", [r for r in rows if r["volume_mm3"] < cuts[0]]),
              (f"medium ({cuts[0]:.1f}-{cuts[1]:.1f} mm3)", [r for r in rows if cuts[0] <= r["volume_mm3"] < cuts[1]]),
              (f"large  (>= {cuts[1]:.1f} mm3)", [r for r in rows if r["volume_mm3"] >= cuts[1]]),
              (f"contrast below {contrast_median:.3f}", [r for r in with_contrast if r["contrast"] < contrast_median]),
              (f"contrast from  {contrast_median:.3f}", [r for r in with_contrast if r["contrast"] >= contrast_median])]
    out["by_size_and_contrast"] = {}
    for label, group in groups:
        entry = dict(n=len(group), sensitivity={n: round(sensitivity(group, n), 3) for n in names})
        out["by_size_and_contrast"][label] = entry
        print(f"{label:28s} n={len(group):4d}  " + "  ".join(f"{n} {entry['sensitivity'][n]:.2f}" for n in names))

    infra = [r for r in rows if r["location"] == "infratentorial"]
    lobar = [r for r in rows if r["location"] == "lobar"]
    first = names[0]

    # ---- 2  infratentorial by structure
    print("\n2  infratentorial lesions by structure")
    out["infratentorial_by_structure"] = {}
    for structure in sorted({r["structure"] for r in infra if r["structure"]}):
        here = [r for r in infra if r["structure"] == structure]
        entry = dict(n=len(here), sensitivity={n: round(sensitivity(here, n), 3) for n in names},
                     median_volume_mm3=round(median([r["volume_mm3"] for r in here]), 2),
                     median_contrast=round(median([r["contrast"] for r in here]), 3))
        out["infratentorial_by_structure"][structure] = entry
        print(f"{structure:26s} n={len(here):4d}  " + "  ".join(f"{n} {entry['sensitivity'][n]:.2f}" for n in names)
              + f"  median {entry['median_volume_mm3']:.2f} mm3, contrast {entry['median_contrast']:.3f}")

    # ---- 3  missed against detected, and how close the net was
    print(f"\n3  infratentorial: missed against detected  (run {first})")
    out["infratentorial_missed_vs_detected"] = {}
    for label, flag in (("detected", True), ("missed", False)):
        here = [r for r in infra if r["runs"][first]["detected"] == flag]
        entry = dict(n=len(here), median_volume_mm3=round(median([r["volume_mm3"] for r in here]), 2),
                     median_contrast=round(median([r["contrast"] for r in here]), 3),
                     median_border_mm=round(median([r["border_mm"] for r in here]), 1),
                     more_than_half_outside_mask=sum(r["outside_fraction"] > 0.5 for r in here))
        out["infratentorial_missed_vs_detected"][label] = entry
        print(f"{label:9s} n={entry['n']:4d}  median {entry['median_volume_mm3']:6.2f} mm3  contrast {entry['median_contrast']:.3f}  "
              f"border {entry['median_border_mm']:.1f} mm  >half outside mask {entry['more_than_half_outside_mask']}")
    out["highest_probability_on_missed"] = {}
    for location, group in (("infratentorial", infra), ("lobar", lobar)):
        missed = [r["runs"][first]["p_max"] for r in group if not r["runs"][first]["detected"]]
        entry = dict(n=len(missed), below_0_05=sum(p < 0.05 for p in missed), from_0_05_to_threshold=sum(0.05 <= p <= THRESHOLD for p in missed),
                     above_threshold_but_removed=sum(p > THRESHOLD for p in missed))
        out["highest_probability_on_missed"][location] = entry
        print(f"highest probability on the {entry['n']} missed {location} lesions: <0.05 {entry['below_0_05']}, "
              f"0.05-{THRESHOLD} {entry['from_0_05_to_threshold']}, >{THRESHOLD} but removed by the size rule {entry['above_threshold_but_removed']}")

    # ---- 4  place or lesion?  strata of equal size and contrast
    print(f"\n4  place or lesion?  sensitivity within strata of equal size and contrast  (run {first})")
    usable = [r for r in rows if r["contrast"] is not None]
    size_cuts = np.percentile([r["volume_mm3"] for r in usable], [33.3, 66.7])
    contrast_cut = float(np.median([r["contrast"] for r in usable]))
    stratum = lambda r: (int(np.searchsorted(size_cuts, r["volume_mm3"], side="right")), int(r["contrast"] >= contrast_cut))
    expected, counted = 0.0, 0
    out["strata"] = dict(size_cuts_mm3=[round(float(c), 2) for c in size_cuts], contrast_cut=round(contrast_cut, 3), cells=[])
    print(f"size cuts {size_cuts[0]:.2f} / {size_cuts[1]:.2f} mm3, contrast cut {contrast_cut:.3f}")
    for s in range(3):
        for k in range(2):
            lo = [r for r in lobar if r["contrast"] is not None and stratum(r) == (s, k)]
            inf = [r for r in infra if r["contrast"] is not None and stratum(r) == (s, k)]
            cell = dict(size=("small", "medium", "large")[s], contrast=("low", "high")[k], lobar_n=len(lo),
                        lobar_sensitivity=round(sensitivity(lo, first), 3) if lo else None, infratentorial_n=len(inf),
                        infratentorial_sensitivity=round(sensitivity(inf, first), 3) if inf else None)
            out["strata"]["cells"].append(cell)
            if lo and inf:
                expected += len(inf) * sensitivity(lo, first); counted += len(inf)
            print(f"{cell['size']:7s} {cell['contrast']:5s}  lobar {sensitivity(lo, first):.2f} (n={len(lo):3d})   "
                  f"infratentorial {sensitivity(inf, first):.2f} (n={len(inf):3d})")
    if counted:                                        # needs lobar AND infratentorial lesions, i.e. a tissue map
        observed = sensitivity([r for r in infra if r["contrast"] is not None], first)
        out["strata"].update(infratentorial_observed=round(observed, 3), infratentorial_expected_at_lobar_rates=round(expected / counted, 3))
        print(f"infratentorial sensitivity observed {observed:.2f}; expected at lobar rates for the same size and contrast {expected / counted:.2f}")
    else:
        print("no stratum holds both lobar and infratentorial lesions (no tissue map?) -- standardisation skipped")

    # ---- 5  a few cases?
    print(f"\n5  a few cases?  (run {first})")
    per_case: Dict[str, List[Dict]] = {}
    for r in infra:
        per_case.setdefault(r["case"], []).append(r)
    ranked = sorted(per_case.values(), key=lambda group: -sum(not r["runs"][first]["detected"] for r in group))
    missed_total = sum(not r["runs"][first]["detected"] for r in infra)
    top = [r for group in ranked[:3] for r in group]; rest = [r for group in ranked[3:] for r in group]
    top_missed = sum(not r["runs"][first]["detected"] for r in top)
    out["case_concentration"] = dict(cases_with_infratentorial_lesions=len(per_case), top3_lesions=len(top), top3_missed=top_missed,
                        missed_total=missed_total, sensitivity_without_top3=round(sensitivity(rest, first), 3))
    print(f"{len(per_case)} cases carry infratentorial lesions; the 3 cases with the most misses hold {len(top)} of {len(infra)} lesions and "
          f"{top_missed} of {missed_total} misses; sensitivity without them {sensitivity(rest, first):.2f}")
    top_ids = {group[0]["case"] for group in ranked[:3]}
    print(f"those 3 cases hold {sum(r['case'] in top_ids for r in rows)} of all {len(rows)} lesions; small = below {size_cuts[0]:.2f} mm3")
    out["case_concentration"]["by_location_without_top3"] = {}
    for location in ENGLISH.values():
        everyone = [r for r in rows if r["location"] == location]
        others = [r for r in everyone if r["case"] not in top_ids]
        inside = [r for r in everyone if r["case"] in top_ids]
        if not everyone:
            continue
        small = lambda group: int(sum(r["volume_mm3"] < size_cuts[0] for r in group))
        entry = dict(n_others=len(others), sensitivity_others=round(sensitivity(others, first), 3) if others else None, small_others=small(others),
                     n_top3=len(inside), sensitivity_top3=round(sensitivity(inside, first), 3) if inside else None, small_top3=small(inside))
        out["case_concentration"]["by_location_without_top3"][location] = entry
        print(f"{location:16s} other cases: n={len(others):4d} sens {sensitivity(others, first):.2f} small {small(others):3d}   |   "
              f"the 3 cases: n={len(inside):4d} sens {sensitivity(inside, first):.2f} small {small(inside):3d}")
    if args.case_list:
        listing = [dict(case=group[0]["case"], infratentorial=len(group), missed=sum(not r["runs"][first]["detected"] for r in group),
                        all_lesions=sum(r["case"] == group[0]["case"] for r in rows),
                        median_volume_mm3=round(median([r["volume_mm3"] for r in group]), 2)) for group in ranked]
        json.dump(dict(run=first, note="PRIVATE -- case ids of the private cohort", cases=listing), open(args.case_list, "w"), indent=1)

    # ---- 6  lower threshold in the infratentorial region only (exploratory)
    print("\n6  EXPLORATORY: lower threshold inside the infratentorial region only")
    out["region_threshold"] = {}
    is_infra = [r["location"] == "infratentorial" for r in rows]
    for name in names:
        out["region_threshold"][name] = {}
        for t in REGION_THRESHOLDS:
            fp = sum(r["what_if"][name][t]["fp"] for r in results)
            detected = [d for r in results for d in r["what_if"][name][t]["detected"]]
            tp = sum(detected); infra_sens = sum(d for d, i in zip(detected, is_infra) if i) / max(sum(is_infra), 1)
            entry = dict(f1=round(f1(tp, fp, len(detected) - tp), 3), sensitivity=round(tp / len(detected), 3),
                         fp_per_case=round(fp / len(results), 2), infratentorial_sensitivity=round(infra_sens, 3))
            out["region_threshold"][name][str(t)] = entry
            print(f"{name:14s} threshold there {t:<4}  F1 {entry['f1']:.3f}  sensitivity {entry['sensitivity']:.2f}  "
                  f"FP/case {entry['fp_per_case']:.2f}  infratentorial sensitivity {entry['infratentorial_sensitivity']:.2f}")
    if args.json:                                     # numpy scalars are not JSON serialisable: convert instead of crashing at the very end
        json.dump(out, open(args.json, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
