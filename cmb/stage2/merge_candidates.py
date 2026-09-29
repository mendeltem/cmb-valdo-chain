"""Merge the candidate sets of several first stages into their UNION -- one candidate per location.

Why (2026-09-22, WORKLOG.md 5w): the second stage can only reject candidates, never invent a missed lesion, so the ceiling of
the whole chain is the candidate step. Measured with ``cmb.analysis.candidate_ceiling``: three seeds of the SAME recipe reach
112 / 103 / 112 of 139 reference lesions at threshold 0.05 -- but their UNION reaches 120, because different seeds miss
different lesions. Averaging the probability maps (``cmb.analysis.ensemble``) cannot use this: the mean damps what only one
member saw. On the candidate level the right operator is the union, because the second stage is there to sort out the extra
false candidates anyway.

What "one candidate per location" means: candidates of different runs are the same location if their stored patches overlap
in space, i.e. their centres are closer than ``--radius`` mm in every axis (default 3 mm, about one microbleed diameter).
Such a group is represented by its member with the HIGHEST stage-1 probability -- its patch, its size, its tissue label. Two
consequences to keep in mind: (1) the stored probability channel of the patch is that of the winning run, not a mean; (2) the
``touched`` list of the group is the union of its members', which is the correct bookkeeping for the lesion-level metric.

Each merged candidate carries the extra field ``n_runs`` (how many input sets found it). That is also the raw material for
the seed-agreement analysis of false positives (``cmb.analysis.label_noise_proxies``).

    python -m cmb.stage2.merge_candidates --input ergebnisse/stage2/union-s42-t005 ergebnisse/stage2/union-s1-t005 \
        ergebnisse/stage2/union-s2-t005 --out ergebnisse/stage2/union-3seeds-t005

CPU only. Private cohorts must get a private --out (patches are image data).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

ROOT = "/home/uchralt/data/work/valdo-t2s"
VOXEL_MM = np.array([0.5, 0.5, 1.0])        # centre_voxel is stored in the volume order (x, y, z) of cmb.stage2.candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="two or more candidate folders (candidates.json + patches.npy)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--radius", type=float, default=3.0, help="mm; candidates closer than this in every axis are one location")
    parser.add_argument("--min-runs", type=int, default=1,
                        help="keep only locations that at least this many input runs found. 1 = the full union. Measured on VALDO "
                             "(2026-09-23): 1 -> 703 candidates, 581 false, 120 lesions reached; 2 -> 325 / 213 / 110; 3 -> 209 / 109 / 98. "
                             "The full union LOWERED the chain (-0.039, WORKLOG.md 5y): the 378 locations only one run sees are mostly that "
                             "run's own errors. min-runs 2 keeps most of the ceiling for a candidate count the classifier handles.")
    args = parser.parse_args()
    if len(args.input) < 2:
        raise SystemExit("--input needs at least two candidate folders")
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    out = absolute(args.out)
    os.makedirs(out, exist_ok=True)

    sets = []
    for folder in args.input:
        folder = absolute(folder)
        info = json.load(open(f"{folder}/candidates.json"))
        sets.append((os.path.basename(folder), info, np.load(f"{folder}/patches.npy", mmap_mode="r")))
        print(f"{os.path.basename(folder)}: {len(info['candidates'])} candidates, "
              f"{sum(bool(r['touched']) for r in info['candidates'])} on a microbleed", flush=True)
    base = sets[0][1]
    for name, info, _ in sets[1:]:
        if info["cases"] != base["cases"]:
            raise SystemExit(f"{name}: different cases or folds than {sets[0][0]} -- the sets must be comparable")
        if info["threshold"] != base["threshold"]:
            print(f"WARNING: {name} has threshold {info['threshold']}, {sets[0][0]} has {base['threshold']}", flush=True)

    # group per case: greedy, highest probability first -- so the representative of a group is always its strongest member
    by_case: Dict[str, List[tuple]] = defaultdict(list)
    for s, (name, info, patches) in enumerate(sets):
        for i, row in enumerate(info["candidates"]):
            by_case[row["case"]].append((-row["probability"], s, i, row))

    rows: List[Dict] = []
    keep_patches: List[np.ndarray] = []
    groups_total = merged_total = 0
    for case in sorted(by_case):
        entries = sorted(by_case[case], key=lambda e: (e[0], e[1], e[2]))
        centres: List[np.ndarray] = []
        members: List[List[tuple]] = []
        for _, s, i, row in entries:
            centre = np.asarray(row["centre_voxel"], float) if "centre_voxel" in row else None
            if centre is None:
                raise SystemExit("the candidate files carry no centre_voxel -- rebuild with the current candidates.py")
            hit = None
            for g, c in enumerate(centres):
                if np.all(np.abs((centre - c) * VOXEL_MM) <= args.radius):
                    hit = g
                    break
            if hit is None:
                centres.append(centre); members.append([(s, i, row)])
            else:
                members[hit].append((s, i, row))
        for group in members:
            s, i, row = group[0]                                   # strongest member = representative
            n_runs = len({m[0] for m in group})
            if n_runs < args.min_runs:
                continue
            touched = sorted({t for _, _, r in group for t in r["touched"]})
            rows.append(dict(row, touched=touched, n_runs=n_runs, runs=sorted({sets[m[0]][0] for m in group})))
            keep_patches.append(np.asarray(sets[s][2][i]))
            groups_total += 1
            merged_total += len(group) - 1

    np.save(f"{out}/patches.npy", np.stack(keep_patches))
    json.dump(dict(threshold=base["threshold"], min_mm3=base["min_mm3"], half=base.get("half"),
                   merged_from=[n for n, _, _ in sets], radius_mm=args.radius, min_runs=args.min_runs,
                   candidates=rows, cases=base["cases"]), open(f"{out}/candidates.json", "w"))

    positives = sum(bool(r["touched"]) for r in rows)
    reachable = sum(len({t for r in rows if r["case"] == c for t in r["touched"]}) for c in base["cases"])
    total = sum(v["n_reference"] for v in base["cases"].values())
    einzeln = [sum(1 for r in rows if r["n_runs"] == k) for k in range(1, len(sets) + 1)]
    print(f"\n{groups_total} candidates after merging ({merged_total} duplicate mentions removed); "
          f"{positives} on a microbleed, {groups_total - positives} false")
    print("found by exactly k runs: " + ", ".join(f"k={k}: {n}" for k, n in enumerate(einzeln, 1)))
    print(f"reached {reachable} of {total} reference microbleeds = upper bound of sensitivity {reachable / total:.3f}")
    print("written to", out)


if __name__ == "__main__":
    main()
