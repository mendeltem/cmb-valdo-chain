"""Queue driver: run ALL phase-4 blocks of one (mod, fold) sequentially on the GPU.

Block list per (mod, fold), STRAT draws only (default draw used everywhere):
  S     n in 3,5,10,15 (3 seeds) + 20,30,40 (2 seeds) + all (1)   = 19
  FT    n in 3,5,10,15 (3 seeds) + 20,30,40 (2 seeds) + all (1)   = 19
  FTn   n in 5,10 (3 seeds) + 20 (2 seeds)                        = 8
  n = 0 is NOT a training block: it is the VALDO model as is (measured, phase4_eval_*_n0.json).

Total per (mod, fold): 46 STRAT blocks + 32 NONE/LOW/MID/HIGH blocks (FT, n = 5, 10, 20; added 31.10.2026; RAND draws exist but are not run) = 78. Each block trains (<= ~15 min
at n=all, less at small n) and predicts one test fold (~5 min). The blocks are resumable:
a finished block (meta.json ok) is skipped.

Usage:  python queue_driver.py --mod t2star --fold 0
"""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import argparse
import json
import os
import subprocess
import sys
import time

PY = _PY
HERE = os.path.dirname(os.path.abspath(__file__))


def blocks_for(mod, fold):
    out = []
    for variant, ns in (("S", [3, 5, 10, 15, 20, 30, 40, "all"]),
                        ("FT", [3, 5, 10, 15, 20, 30, 40, "all"]),
                        ("FTn", [5, 10, 20])):
        for n in ns:
            seeds = [42] if n == "all" else ([42, 1, 7] if n <= 15 else [42, 1])
            for seed in seeds:
                out.append((variant, str(n), seed, "STRAT"))
    # 31.10.2026: the draw-strategy arm of the wk0b spec (user 30.10. "stratified vs randomized"), FT only,
    # n = 5, 10, 20, same seeds as STRAT; it was missing from the driver.
    for strategy in ("NONE", "LOW", "MID", "HIGH"):      # 31.10.2026 (user): burden spectrum; RAND dropped ("more like
                                                        # the stratified mix" = STRAT, the default draw of every block)
        for n in (5, 10, 20):
            for seed in ([42, 1, 7] if n <= 15 else [42, 1]):
                out.append(("FT", str(n), seed, strategy))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", required=True, choices=["t2star", "swi"])
    ap.add_argument("--fold", type=int, required=True, choices=range(5))
    ap.add_argument("--only", help="comma list of VARIANT_n (e.g. FT_5,FTn_5) for debugging")
    a = ap.parse_args()
    only = None
    if a.only:
        only = set(a.only.split(","))
    blocks = blocks_for(a.mod, a.fold)
    if only:
        blocks = [b for b in blocks if f"{b[0]}_{b[1]}" in only or f"{b[0]}_{b[1]}_{b[3]}" in only]
    log = open(f"{HERE}/driver_{a.mod}_fold{a.fold}.log", "a")
    t0 = time.time()
    for i, (variant, n, seed, strategy) in enumerate(blocks, 1):
        outd = f"{HERE}/unet/{a.mod}/{variant}_n{n}_fold{a.fold}_s{seed}" + ("" if strategy == "STRAT" else f"_{strategy}")
        done = os.path.exists(f"{outd}/meta.json") and json.load(open(f"{outd}/meta.json")).get("ok")
        if done:
            print(f"[{i}/{len(blocks)}] {variant} n={n} s{seed} {strategy}: already done -- skipped", flush=True)
            continue
        cmd = [PY, f"{HERE}/run_block.py", "--mod", a.mod, "--fold", str(a.fold),
               "--variant", variant, "--n", n, "--seed", str(seed), "--strategy", strategy]
        print(f"[{i}/{len(blocks)}] {variant} n={n} s{seed} {strategy} START {time.time()-t0:.0f}s", flush=True)
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[{i}/{len(blocks)}] {variant} n={n} s{seed} {strategy} FAILED rc={r.returncode} -- "
                  f"continuing with the next block (see {HERE}/driver_{a.mod}_fold{a.fold}.log)",
                  flush=True)
    print(f"DRIVER DONE {a.mod} fold {a.fold}: {len(blocks)} blocks in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
