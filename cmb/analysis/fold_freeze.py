"""Freeze the 5-fold assignment of every cohort and check that every input of a run carries exactly those folds (2026-10-25).

User 2026-10-25: "the 5-fold CV is now fixed, whatever we do, so that everything is repeatable for VALDO and in-house?" The folds
already live in files (VALDO: dev/split.json from code/split.py, seed 0, stratified by cohort x burden, unchanged since 2026-09-11;
in-house cohorts: the 'fold' field of their manifests; the SWI twins of T2* subjects are moved to the T2* fold for stage 2 by
manifest_swi_mix.json). Nothing recomputes them -- but nothing guarded them either. This tool
  freeze  writes <out>/<cohort>.json ({case: fold}) plus <out>/<cohort>.sha256 from a reference source; refuses to overwrite a
          different existing freeze;
  check   compares candidate folders (cases in candidates.json), prediction runs (fold<k>/<id>_pred_proba.nii.gz) and manifests
          against a freeze and prints ONLY counts; exit 1 on any mismatch or on a changed freeze file (sha256).

    python -m cmb.analysis.fold_freeze freeze --cohort valdo --split dev/split.json --out dev/folds_frozen
    python -m cmb.analysis.fold_freeze freeze --cohort swi_stage2 --manifest <MB>/gitter/manifest_swi_mix.json --out <MB>/folds_frozen
    python -m cmb.analysis.fold_freeze check --frozen dev/folds_frozen/valdo.json --candidates ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 \
        --runs ergebnisse/turnier/a03-aniso-e60-gd --manifests dev/manifest_valdo_gitter_d.json
Private cohorts: private --out (the freeze lists case ids).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Dict

from cmb.paths import ROOT


def absolute(p: str) -> str:
    return p if os.path.isabs(p) else f"{ROOT}/{p}"


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def folds_from_split(path: str) -> Dict[str, int]:
    return {cid: k for k, ids in enumerate(json.load(open(absolute(path)))["folds"]) for cid in ids}


def folds_from_manifest(path: str) -> Dict[str, int]:
    return {e["id"]: int(e["fold"]) for e in json.load(open(absolute(path))) if int(e["fold"]) in range(5)}


def folds_from_candidates(folder: str) -> Dict[str, int]:
    return {c: int(v["fold"]) for c, v in json.load(open(f"{absolute(folder)}/candidates.json"))["cases"].items()}


def folds_from_run(folder: str) -> Dict[str, int]:
    out = {}
    for p in glob.glob(f"{absolute(folder)}/fold*/*_pred_proba.nii.gz"):
        out[os.path.basename(p)[:-len("_pred_proba.nii.gz")]] = int(p.split("/fold")[-1][0])
    return out


def freeze(args) -> int:
    folds = folds_from_split(args.split) if args.split else folds_from_manifest(args.manifest)
    out = absolute(args.out); os.makedirs(out, exist_ok=True)
    path = f"{out}/{args.cohort}.json"
    text = json.dumps(dict(cohort=args.cohort, source=args.split or args.manifest, folds=dict(sorted(folds.items()))), indent=1)
    if os.path.exists(path):
        old = json.load(open(path))
        if old["folds"] != dict(sorted(folds.items())):
            print(f"REFUSED: {path} exists with a DIFFERENT assignment -- folds must not change; nothing written")
            return 1
        print(f"{args.cohort}: freeze exists and matches ({len(folds)} cases)")
        return 0
    open(path, "w").write(text)
    open(f"{out}/{args.cohort}.sha256", "w").write(f"{sha256(path)}  {os.path.basename(path)}\n")
    sizes = [sum(v == k for v in folds.values()) for k in range(5)]
    print(f"{args.cohort}: frozen {len(folds)} cases, fold sizes {sizes} -> {path}")
    return 0


def check(args) -> int:
    path = absolute(args.frozen)
    sums = path[:-5] + ".sha256"
    bad = 0
    if not os.path.exists(sums) or open(sums).read().split()[0] != sha256(path):
        print(f"FREEZE CHANGED OR UNSIGNED: {path}"); bad += 1
    ref = json.load(open(path))["folds"]
    sources = [("candidates", f, folds_from_candidates) for f in args.candidates] + [("run", f, folds_from_run) for f in args.runs] \
        + [("manifest", f, folds_from_manifest) for f in args.manifests]
    for kind, f, reader in sources:
        try:
            got = reader(f)
        except Exception as e:
            print(f"{kind:10s} {f}: unreadable ({type(e).__name__})"); bad += 1; continue
        common = set(got) & set(ref)
        mismatch = sum(got[c] != ref[c] for c in common)
        unknown = len(set(got) - set(ref))
        bad += mismatch + unknown + (len(common) == 0)
        print(f"{kind:10s} {f}: {len(got)} cases, {mismatch} fold mismatches, {unknown} not in the freeze")
    print("FOLDS OK" if not bad else f"FOLD PROBLEMS: {bad}")
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--cohort", required=True); f.add_argument("--split"); f.add_argument("--manifest")
    f.add_argument("--out", required=True)
    c = sub.add_parser("check"); c.add_argument("--frozen", required=True)
    c.add_argument("--candidates", nargs="*", default=[]); c.add_argument("--runs", nargs="*", default=[]); c.add_argument("--manifests", nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "freeze" and bool(args.split) == bool(args.manifest):
        raise SystemExit("give exactly one of --split and --manifest")
    sys.exit(freeze(args) if args.command == "freeze" else check(args))


if __name__ == "__main__":
    main()
