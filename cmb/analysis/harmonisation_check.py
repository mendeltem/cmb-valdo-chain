"""Which intensity normalisation actually brings the cohorts together? -- measured, without training anything.

Why (2026-09-23, user question): transfer between our cohorts fails on the IMAGES, not on the annotation style
(WORKLOG.md 5n). The brain level differs (0.52 against 0.71 on the inverted scale), the lesion contrast differs, and
the percentile scale p99.5 turned out not to close the gap (5aa) -- three CPU minutes that saved two GPU hours.
The same pre-check can be run for every other candidate before anyone trains a network with it.

What "brings together" means here, and why these four numbers: a network trained on cohort A and applied to cohort B
sees the image. It can only transfer if, after normalisation,

  * the brain sits at the same LEVEL (median) -- otherwise every learned threshold is off;
  * the brain has the same SPREAD (interquartile range) -- otherwise "clearly darker than the surroundings" means
    something different;
  * a microbleed has the same CONTRAST against its own surroundings (lesion minus a ring 1-3 mm around it);
  * a microbleed sits at the same absolute LEVEL.

Reported per normalisation: those four numbers per cohort and, as the summary, the SPREAD between cohorts
(max - min). Smaller spread = better harmonised. The spread of the lesion contrast is the one that matters most:
it is what the network detects.

Normalisations (all computed inside the brain mask, per case):
    max         image / max                     -- what we use today, the original's choice; one bright outlier moves everything.
                                                   Note this is NOT a 0-1 scale: the brain ends up somewhere in 0.5-0.98, the
                                                   lower half of the range is never used
    p995        image / 99.5th percentile       -- outlier-robust version of the same idea
    minmax      (image - min) / (max - min)     -- the textbook 0-1 scale; the brain fills [0, 1] exactly in every case, but
                                                   TWO single voxels (the darkest and the brightest) define the whole scale
    p1_p99      (image - P1) / (P99 - P1)       -- the robust 0-1 scale: percentiles instead of extremes, clipped to [0, 1]
    z           (image - mean) / std            -- aligns level AND spread, but discards the absolute scale
    median_iqr  (image - median) / IQR          -- the robust version of z (lesions and artefacts do not move the anchors)
    wm          image / median of white matter  -- anchored to TISSUE instead of to a distribution (needs SynthSeg;
                                                   the classic "white-stripe" idea: normal-appearing white matter is
                                                   the one thing every brain has in comparable amounts)
    landmarks   piecewise-linear mapping of the 10th/25th/50th/75th/90th/99th percentile onto those of a reference
                cohort -- the Nyul-Udupa scheme; the only candidate here that can fix a NON-linear difference

    python -m cmb.analysis.harmonisation_check --cohort valdo=dev/manifest_valdo_gitter_d.json \\
        --synthseg valdo="dev/synthseg/{id}_synthseg.nii.gz" --reference valdo --cases 20

CPU only, a few minutes. Private cohorts: only the summary numbers leave this script, never a case.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
CONNECTIVITY_26 = np.ones((3, 3, 3), bool)
WM_LABELS = {2, 41}                      # cerebral white matter left / right (FreeSurfer table)
LANDMARKS = (10, 25, 50, 75, 90, 99)


def normalise(kind: str, image: np.ndarray, brain: np.ndarray, wm: np.ndarray, referenz) -> np.ndarray:
    werte = image[brain]
    if kind == "max":
        return image / max(werte.max(), 1e-6)
    if kind == "p995":
        return image / max(np.percentile(werte, 99.5), 1e-6)
    if kind == "minmax":                     # a real 0-1 scale: darkest brain voxel -> 0, brightest -> 1
        lo, hi = werte.min(), werte.max()
        return (image - lo) / max(hi - lo, 1e-6)
    if kind == "p1_p99":                     # robust 0-1 scale: not the extreme values but the 1st and 99th percentile
        lo, hi = np.percentile(werte, [1, 99])
        return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if kind == "z":
        return (image - werte.mean()) / max(werte.std(), 1e-6)
    if kind == "median_iqr":
        q1, q3 = np.percentile(werte, [25, 75])
        return (image - np.median(werte)) / max(q3 - q1, 1e-6)
    if kind == "wm":
        anker = np.median(image[wm]) if wm.any() else np.median(werte)
        return image / max(anker, 1e-6)
    if kind == "landmarks":
        quellen = np.percentile(werte, LANDMARKS)
        return np.interp(image, quellen, referenz)          # outside the landmarks: continued constant
    raise ValueError(kind)


def bloecke(label: np.ndarray) -> List:
    """Bounding box per microbleed, once in advance -- the ring is computed ONLY inside it.

    Without this, one dilates per microbleed and per normalisation over the whole volume (292 x 341 x 140): with three cohorts
    and six methods that is thousands of full-volume dilations, and the run takes hours instead of minutes
    (done myself and aborted after 42 min, 2026-09-23 -- the same mistake as the day before in label_noise_proxies).
    """
    l, n = ndimage.label(label, structure=CONNECTIVITY_26)
    aus = []
    for box in ndimage.find_objects(l):
        weit = tuple(slice(max(b.start - 7, 0), min(b.stop + 7, s)) for b, s in zip(box, label.shape))
        aus.append((weit, l[weit]))
    return [(w, (teil == j)) for j, (w, teil) in enumerate(aus, 1)]


def kennzahlen(bild: np.ndarray, brain: np.ndarray, blocks: List) -> List[Dict]:
    """-> per microbleed: brain level, brain spread, lesion level, contrast against a ring 1-3 mm around it."""
    werte = bild[brain]
    q1, q3 = np.percentile(werte, [25, 75])
    niveau, streuung = float(np.median(werte)), float(q3 - q1)
    zeilen = []
    for weit, m in blocks:
        innen = ndimage.binary_dilation(m, CONNECTIVITY_26, 2)
        ring = ndimage.binary_dilation(m, CONNECTIVITY_26, 6) & ~innen & brain[weit]
        if not ring.any():
            continue
        zeilen.append(dict(niveau=niveau, streuung=streuung, laesion=float(bild[weit][m].mean()),
                           kontrast=float(bild[weit][m].mean() - bild[weit][ring].mean())))
    return zeilen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", nargs="+", required=True, help="NAME=MANIFEST")
    parser.add_argument("--synthseg", nargs="*", default=[], help="NAME=PATTERN with {id} or {sid}; without it 'wm' is skipped")
    parser.add_argument("--reference", help="Cohort onto whose landmarks the mapping is done (default: the first)")
    parser.add_argument("--cases", type=int, default=20, help="Cases per cohort (the first ones with a microbleed)")
    parser.add_argument("--json")
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    kohorten = dict(spec.split("=", 1) for spec in args.cohort)
    muster = dict(spec.split("=", 1) for spec in args.synthseg)
    referenz_name = args.reference or list(kohorten)[0]

    # 1) load the data once
    geladen: Dict[str, List] = {}
    for name, manifest in kohorten.items():
        eintraege = json.load(open(absolute(manifest)))
        genommen = []
        for e in eintraege:
            if len(genommen) >= args.cases:
                break
            cid, d = e["id"], e["dir"]
            label = np.asarray(nib.load(f"{d}/{cid}_label.nii.gz").dataobj) > 0
            if not label.any():
                continue
            bild = np.asarray(nib.load(f"{d}/{cid}_image.nii.gz").dataobj).astype(np.float32)
            brain = bild > 0
            wm = np.zeros_like(brain)
            if name in muster:
                from cmb.analysis.fp_location import load_synthseg, synthseg_path
                pfad = synthseg_path(muster[name], cid)
                if os.path.exists(pfad):
                    seg = load_synthseg(pfad, nib.load(f"{d}/{cid}_image.nii.gz"))
                    wm = np.isin(seg, list(WM_LABELS)) & brain
            genommen.append((bild, brain, wm, bloecke(label)))   # blocks once per case, not per normalisation
        geladen[name] = genommen
        print(f"{name}: {len(genommen)} cases with a microbleed loaded"
              + ("" if name in muster else "  (no tissue map -- 'wm' is skipped)"), flush=True)

    # 2) landmarks of the reference cohort
    referenz = np.median([np.percentile(b[br], LANDMARKS) for b, br, _, _ in geladen[referenz_name]], axis=0)

    verfahren = ["max", "p995", "minmax", "p1_p99", "z", "median_iqr", "landmarks"] + (["wm"] if muster else [])
    aus: Dict = dict(cohorts=list(kohorten), reference=referenz_name, cases=args.cases, methods={})
    print(f"\nLandmarks of the reference ({referenz_name}, percentiles {LANDMARKS}): " + " ".join(f"{v:.3f}" for v in referenz))
    print(f"\n{'method':12s} {'cohort':16s} {'brain level':>12s} {'brain spread':>14s} {'lesion':>9s} {'contrast':>9s}")
    for kind in verfahren:
        je_kohorte = {}
        for name, faelle in geladen.items():
            zeilen = []
            for bild, brain, wm, blocks in faelle:
                if kind == "wm" and not wm.any():
                    continue
                zeilen += kennzahlen(normalise(kind, bild, brain, wm, referenz), brain, blocks)
            if not zeilen:
                continue
            a = {s: float(np.median([z[s] for z in zeilen])) for s in ("niveau", "streuung", "laesion", "kontrast")}
            a["n"] = len(zeilen)
            je_kohorte[name] = a
            print(f"{kind:12s} {name:16s} {a['niveau']:12.3f} {a['streuung']:14.3f} {a['laesion']:9.3f} {a['kontrast']:9.3f}")
        spanne = {s: max(v[s] for v in je_kohorte.values()) - min(v[s] for v in je_kohorte.values())
                  for s in ("niveau", "streuung", "laesion", "kontrast")}
        aus["methods"][kind] = dict(per_cohort=je_kohorte, spread=spanne)
        print(f"{'':12s} {'-> spread':16s} {spanne['niveau']:12.3f} {spanne['streuung']:14.3f} {spanne['laesion']:9.3f} {spanne['kontrast']:9.3f}\n")

    print("Smaller spread = better harmonised. The CONTRAST column is the most important one: it is what the network detects.")
    bester = min(aus["methods"], key=lambda k: aus["methods"][k]["spread"]["kontrast"])
    print(f"Smallest contrast spread: {bester} ({aus['methods'][bester]['spread']['kontrast']:.3f}); used today: max "
          f"({aus['methods']['max']['spread']['kontrast']:.3f}).")
    if args.json:
        json.dump(aus, open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
