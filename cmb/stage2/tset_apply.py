"""Arena 2: confirm ONE carried stage-2 arm on fresh cases (2026-10-25).

cmb.stage2.tset_summary only SELECTS ('carry to confirmation'): with ~190 scans the same cases chose the arm and measured it,
so its delta is optimistic (winner's curse; Arena 2 round 3, R2). The confirmation uses cases that took no part in the selection,
e.g. the external public SWI set. For the arm and for the control, and for every --seed-offset:
  1. threshold: 5-fold cross-validation BY CASE over the whole training pool (the cohorts of the Arena-2 runs) -> out-of-fold
     score of every pool candidate -> ONE threshold maximising the pooled lesion F1 over all pool cases (fixed before the target
     is touched; per-cohort thresholds are recorded but not used);
  2. train on ALL pool candidates, score the target candidates, accept score >= threshold.
Pre-registered test (one, on the target only): paired bootstrap over target cases of the seed-averaged F1 difference arm minus
control; 'confirmed' if the 2.5th percentile is above 0 (one-sided 2.5 %). Reference rows on the target: the stage-1 probability
at the project's fixed cell (0.3, >= 2 mm3) with and without the hard CSF rule. Nothing of the target is used for any choice.

    python -m cmb.stage2.tset_apply --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 t2star=<MB>/stage2/...-z40 \
        swi=<MB>/stage2/...-z40 --folds swi=<MB>/gitter/manifest_swi_mix.json --target extern=<folder> --arm tset-track \
        [--control tset-plain] [--seed-offsets 0 1] --json <private>/tset_confirm_<arm>.json
The target candidates must come from cmb.stage2.candidates with the same --half as the pool (track arms: 20 20 20) and carry
'centre_voxel'. Sums only on screen; per-case counts go to the JSON (keep it private for private cohorts).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from cmb.stage2 import trackset as T
from cmb.stage2.classifier import THRESHOLDS
from cmb.stage2.pooled import TISSUE_CLASS, scores_of

from cmb.paths import ROOT
CSF_LABELS = (4, 5, 14, 15, 24, 43, 44)


def absolute(p: str) -> str:
    return p if os.path.isabs(p) else f"{ROOT}/{p}"


def load_sets(specs: List[str], fold_override: Dict[str, Dict[str, int]]):
    rows, blocks, cases = [], [], {}
    for spec in specs:
        name, folder = spec.split("=", 1)
        info = json.load(open(f"{absolute(folder)}/candidates.json"))
        for cid, c in info["cases"].items():
            cases[cid] = dict(fold=fold_override.get(name, {}).get(cid, c["fold"]), n_reference=c["n_reference"], cohort=name)
        for r in info["candidates"]:
            rows.append(dict(r, fold=cases[r["case"]]["fold"], cohort=name))
        blocks.append(np.asarray(np.load(f"{absolute(folder)}/patches.npy", mmap_mode="r")))
    missing = sum("centre_voxel" not in r for r in rows)
    if missing:
        raise SystemExit(f"{missing} candidates without 'centre_voxel': rebuild them with the current cmb.stage2.candidates")
    return rows, np.concatenate(blocks), cases


def bundle_of(rows, patches, need_tracks: bool) -> T.Bundle:
    ids = {c: i for i, c in enumerate(sorted({r["case"] for r in rows}))}
    centre = np.array([r["centre_voxel"] for r in rows], np.float32) * T.VOXEL_MM_XYZ
    tracks = T.track_candidates(patches) if need_tracks else None
    return T.Bundle(patches, T.side_from_rows(rows, patches, TISSUE_CLASS), [ids[r["case"]] for r in rows], centre, tracks)


def per_case(rows, accepted, cases) -> Dict[str, tuple]:
    counts = {c: [set(), 0] for c in cases}
    for r, keep in zip(rows, accepted):
        if keep:
            if r["touched"]:
                counts[r["case"]][0].update(r["touched"])
            else:
                counts[r["case"]][1] += 1
    return {c: (len(found), fp, cases[c]["n_reference"] - len(found)) for c, (found, fp) in counts.items()}


def pooled_f1(triples) -> float:
    tp, fp, fn = (sum(t[i] for t in triples) for i in range(3))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def run_arm(arm: str, pool_rows, pool, pool_cases, target, seed_offset: int, epochs: int, members: int) -> Dict:
    mods = T.ARMS[arm]
    fit = lambda tx, ty, sx, seed: T.fit_predict_trackset(tx, ty, sx, seed=seed, modules=mods, epochs=epochs, members=members)
    y = np.array([1 if r["touched"] else 0 for r in pool_rows]); fold = np.array([r["fold"] for r in pool_rows])
    oof = np.zeros(len(pool_rows), np.float32)
    for k in sorted(set(fold)):
        held = fold == k
        oof[held] = fit(pool[np.nonzero(~held)[0]], y[~held], pool[np.nonzero(held)[0]], 1000 * seed_offset + k)
    n_ref = sum(c["n_reference"] for c in pool_cases.values()); n_cases = len(pool_cases)
    threshold = float(max(THRESHOLDS, key=lambda t: scores_of(pool_rows, oof >= t, n_ref, n_cases)["f1"]))
    per_cohort_threshold = {}
    for name in sorted({c["cohort"] for c in pool_cases.values()}):
        sel = np.array([r["cohort"] == name for r in pool_rows])
        here = [c for c in pool_cases.values() if c["cohort"] == name]
        sub = [r for r, s in zip(pool_rows, sel) if s]
        per_cohort_threshold[name] = float(max(THRESHOLDS, key=lambda t: scores_of(sub, oof[sel] >= t, sum(c["n_reference"] for c in here), len(here))["f1"]))
    scores = fit(pool, y, target["bundle"], 1000 * seed_offset + 99)
    accepted = scores >= threshold
    return dict(threshold=threshold, per_cohort_threshold=per_cohort_threshold,
                pool_out_of_fold=scores_of(pool_rows, oof >= threshold, n_ref, n_cases),
                target=scores_of(target["rows"], accepted, target["n_ref"], len(target["cases"])),
                per_case=per_case(target["rows"], accepted, target["cases"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", nargs="+", required=True, help="training pool: NAME=FOLDER (the cohorts of the Arena-2 runs)")
    parser.add_argument("--folds", nargs="*", default=[], help="NAME=MANIFEST: folds of this cohort from a manifest (as in pooled)")
    parser.add_argument("--target", required=True, help="NAME=FOLDER: fresh candidates, never used in the selection")
    parser.add_argument("--arm", required=True, choices=sorted(T.ARMS))
    parser.add_argument("--control", default="tset-plain", choices=sorted(T.ARMS))
    parser.add_argument("--seed-offsets", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    override = {}
    for spec in args.folds:
        name, manifest = spec.split("=", 1)
        override[name] = {e["id"]: e["fold"] for e in json.load(open(absolute(manifest)))}
    pool_rows, pool_patches, pool_cases = load_sets(args.cohort, override)
    t_name, _ = args.target.split("=", 1)
    t_rows, t_patches, t_cases = load_sets([args.target], {})
    assert not set(t_cases) & set(pool_cases), "target cases overlap the training pool -- not fresh"
    need_tracks = "track" in T.ARMS[args.arm] or "track" in T.ARMS[args.control]
    pool = bundle_of(pool_rows, pool_patches, need_tracks)
    target = dict(rows=t_rows, cases=t_cases, n_ref=sum(c["n_reference"] for c in t_cases.values()), bundle=bundle_of(t_rows, t_patches, need_tracks))
    print(f"pool: {len(pool_rows)} candidates in {len(pool_cases)} cases; target {t_name}: {len(t_rows)} candidates, "
          f"{target['n_ref']} reference lesions in {len(t_cases)} cases", flush=True)
    out = dict(arm=args.arm, control=args.control, target=t_name, seed_offsets=args.seed_offsets, runs={})
    for arm in (args.arm, args.control):
        out["runs"][arm] = {}
        for s in args.seed_offsets:
            r = run_arm(arm, pool_rows, pool, pool_cases, target, s, args.epochs, args.members)
            out["runs"][arm][str(s)] = r
            print(f"{arm:16s} seed offset {s}: threshold {r['threshold']:.2f} (pool OOF F1 {r['pool_out_of_fold']['f1']:.3f})  "
                  f"target F1 {r['target']['f1']:.3f} P {r['target']['precision']:.2f} S {r['target']['sensitivity']:.2f} "
                  f"FP/case {r['target']['fp_per_case']:.2f}", flush=True)
    prob = np.array([r["probability"] for r in t_rows]); vox = np.array([r["voxels"] for r in t_rows])
    csf = np.array([r.get("tissue", -1) in CSF_LABELS for r in t_rows])
    fixed = (prob >= 0.3) & (vox >= 8)
    out["reference"] = {label: scores_of(t_rows, acc, target["n_ref"], len(t_cases)) for label, acc in
                        (("probability", fixed), ("probability+csf", fixed & ~csf))}
    ids = sorted(t_cases)
    A = [[out["runs"][args.arm][str(s)]["per_case"][c] for c in ids] for s in args.seed_offsets]
    B = [[out["runs"][args.control][str(s)]["per_case"][c] for c in ids] for s in args.seed_offsets]
    delta = float(np.mean([pooled_f1(a) - pooled_f1(b) for a, b in zip(A, B)]))
    rng = np.random.RandomState(0); draws = []
    for _ in range(10000):
        ix = rng.randint(len(ids), size=len(ids))
        draws.append(np.mean([pooled_f1([a[j] for j in ix]) - pooled_f1([b[j] for j in ix]) for a, b in zip(A, B)]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out["test"] = dict(delta_f1=round(delta, 4), ci95=[round(float(lo), 4), round(float(hi), 4)],
                       decision="confirmed" if lo > 0 else "not confirmed",
                       rule="paired case bootstrap of the seed-averaged F1 difference, one-sided 2.5 %, fixed 2026-10-25")
    print(f"\n{args.arm} minus {args.control} on {t_name}: {delta:+.3f} [{lo:+.3f}; {hi:+.3f}] -> {out['test']['decision']}")
    for label, r in out["reference"].items():
        print(f"  reference {label:16s} F1 {r['f1']:.3f} P {r['precision']:.2f} S {r['sensitivity']:.2f} FP/case {r['fp_per_case']:.2f}")
    json.dump(out, open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
