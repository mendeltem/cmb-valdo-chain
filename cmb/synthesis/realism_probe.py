"""Arena 2: can a classifier tell inserted objects from real candidates? (classifier two-sample test, 2026-10-25)

Run BEFORE synthetic candidates enter any training (research round 3, R3; our own lesson from 2026-09-22: augmentation must match
the target statistics before it is trained on). Per cohort, on the candidate crops the stage-2 classifier would see (image, FRST,
stage-1 probability):
  A  real microbleed candidates (touching a reference lesion)  vs  candidates at inserted microbleeds;
  B  real false candidates                                     vs  candidates at inserted veins;
  C  POSITIVE CONTROL: real microbleed candidates vs inserted microbleed crops whose image channel was smoothed (sigma 1.5 voxels)
     -- a known-bad fake. If C is not detected, the probe has no power and A/B mean nothing.
Both classes are subsampled to the same size, the small 3D CNN of cmb.stage2.classifier is trained in 5-fold cross-validation
grouped by case, and the AUC of the out-of-fold scores is reported (null 0.5; sd of the accuracy ~ 1 / (2 sqrt(n))).
Gates, fixed 2026-10-25 (proposal; FDA nodule insertion: readers reached AUC 0.52-0.71, Pezeshk 2015):
  pass = AUC_A <= 0.70 and AUC_B <= 0.70 and AUC_C >= 0.85;  'no power' if AUC_C < 0.85;  else fail.

    python -m cmb.synthesis.realism_probe --real swi=<real candidates> --synthetic swi=<synthetic candidates>,<insert_log.json> \
        [--epochs 40] --json <private>/realism_swi.json
Sums only.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from scipy import ndimage

from cmb.stage2.classifier import CROP, fit_predict_cnn

from cmb.paths import ROOT
GRID_MM_XYZ = np.array([0.5, 0.5, 1.0])
MATCH_MM = {"bleed": 3.0, "vein": 4.0}
GATE_AB, GATE_C = 0.70, 0.85


def absolute(p: str) -> str:
    return p if os.path.isabs(p) else f"{ROOT}/{p}"


def load(folder: str):
    info = json.load(open(f"{absolute(folder)}/candidates.json"))
    return info["candidates"], np.load(f"{absolute(folder)}/patches.npy", mmap_mode="r")


def inserted_candidates(rows: List[Dict], log: Dict[str, List[Dict]], kind: str) -> List[int]:
    """Indices of synthetic-run candidates within MATCH_MM of an inserted object of `kind` in the same case."""
    out = []
    for i, r in enumerate(rows):
        objs = [o for o in log.get(r["case"], []) if o["kind"] == kind and (kind != "bleed" or o.get("labelled", True))]
        c = np.array(r["centre_voxel"], float)
        if any(np.linalg.norm((c - np.array(o["centre_voxel"])) * GRID_MM_XYZ) <= MATCH_MM[kind] for o in objs):
            out.append(i)
    return out


def auc(score, y) -> float:
    pos, neg = score[y == 1], score[y == 0]
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def c2st(x0: np.ndarray, g0: List[str], x1: np.ndarray, g1: List[str], rng, epochs: int) -> Dict:
    n = min(len(x0), len(x1))
    if n < 10:
        return dict(n_per_class=n, auc=None, note="fewer than 10 per class")
    i0 = rng.choice(len(x0), n, replace=False); i1 = rng.choice(len(x1), n, replace=False)
    x = np.concatenate([x0[i0], x1[i1]]).astype(np.float32); y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    groups = np.array([g0[i] for i in i0] + [g1[i] for i in i1])
    uniq = rng.permutation(np.unique(groups)); fold_of = {g: k % 5 for k, g in enumerate(uniq)}
    fold = np.array([fold_of[g] for g in groups]); score = np.zeros(len(y), np.float32)
    for k in range(5):
        te = fold == k
        if te.any() and (~te).any() and len(set(y[~te])) == 2:
            score[te] = fit_predict_cnn(x[~te], y[~te], x[te], seed=k, epochs=epochs)
    return dict(n_per_class=int(n), auc=round(auc(score, y), 3), accuracy=round(float(((score >= 0.5) == y).mean()), 3),
                null_sd_accuracy=round(1 / (2 * np.sqrt(2 * n)), 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", nargs="+", required=True, help="NAME=FOLDER (real candidates)")
    parser.add_argument("--synthetic", nargs="+", required=True, help="NAME=FOLDER,INSERT_LOG (candidates of the modified copies)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    real = dict(s.split("=", 1) for s in args.real)
    out = {}
    for spec in args.synthetic:
        name, rest = spec.split("=", 1); folder, log_path = rest.split(",", 1)
        rng = np.random.RandomState(0)
        r_rows, r_p = load(real[name]); s_rows, s_p = load(folder)
        log = json.load(open(absolute(log_path)))["objects"]
        crop = lambda p, idx: np.stack([np.asarray(p[i], np.float32) for i in idx]) if idx else np.zeros((0,) + tuple(p.shape[1:]), np.float32)
        real_pos = [i for i, r in enumerate(r_rows) if r["touched"]]; real_neg = [i for i, r in enumerate(r_rows) if not r["touched"]]
        syn_b = inserted_candidates(s_rows, log, "bleed"); syn_v = inserted_candidates(s_rows, log, "vein")
        xb = crop(s_p, syn_b); xc = xb.copy()
        for j in range(len(xc)):
            xc[j, 0] = ndimage.gaussian_filter(xc[j, 0], 1.5)
        res = dict(inserted_bleed_candidates=len(syn_b), inserted_vein_candidates=len(syn_v),
                   bleeds_inserted=sum(o["kind"] == "bleed" and o.get("labelled", True) for v in log.values() for o in v),
                   veins_inserted=sum(o["kind"] == "vein" for v in log.values() for o in v))
        g = lambda rows, idx: [rows[i]["case"] for i in idx]
        res["A_bleeds"] = c2st(crop(r_p, real_pos), g(r_rows, real_pos), xb, g(s_rows, syn_b), rng, args.epochs)
        res["B_veins"] = c2st(crop(r_p, real_neg), g(r_rows, real_neg), crop(s_p, syn_v), g(s_rows, syn_v), rng, args.epochs)
        res["C_control"] = c2st(crop(r_p, real_pos), g(r_rows, real_pos), xc, g(s_rows, syn_b), rng, args.epochs)
        a, b, c = (res[k]["auc"] for k in ("A_bleeds", "B_veins", "C_control"))
        if None in (a, b, c):
            res["gate"] = "too few candidates"
        elif c < GATE_C:
            res["gate"] = "no power (control not detected)"
        else:
            res["gate"] = "pass" if a <= GATE_AB and b <= GATE_AB else "fail"
        res["reach"] = dict(bleeds_found_as_candidates=round(len(syn_b) / max(res["bleeds_inserted"], 1), 3))
        out[name] = res
        print(f"{name}: inserted bleeds {res['bleeds_inserted']} (as candidates {len(syn_b)}), veins {res['veins_inserted']} "
              f"(as candidates {len(syn_v)}); AUC A {a} B {b} control {c} -> {res['gate']}", flush=True)
    json.dump(dict(gates=dict(ab_max=GATE_AB, control_min=GATE_C), crop=list(CROP), cohorts=out), open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
