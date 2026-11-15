"""Nested scan draws for the Phase-2 fine-tuning curves (wk0b / wk4).

For each target modality (in-house T2* and SWI) and each frozen fold k, the n-scan training
sets are NESTED: the 3 scans are inside the 5, the 5 inside the 10, ... . One ordered list of
the training scans per (fold, strategy, seed); the prefix of length n is the training set.
Seeds: 42, 1, 7 for n <= 15 (three draws), 42 and 1 for n >= 20 (two draws, heavy overlap).

Strategies
  STRAT  the default draw: proportional round-robin over the four lesion-count classes
         (0, 1-2, 3-5, >5), so small n still contains the mix of the training fold
  RAND   a purely random permutation (may contain no CMB scan at small n -- the summary
         reports how often)
  LOW    deterministic: scans with >= 1 CMB, fewest CMB first, CMB-free scans last (added 31.10.2026)
  MID    deterministic: scans with >= 1 CMB closest to the median count first (added 31.10.2026)
  NONE   only CMB-free scans, random order per seed; capped at their number (added 31.10.2026)
  HIGH   deterministic: scans with the most CMB first (used with three init seeds; the draw
         itself is the same for all three)

Folds and lesion counts come from the frozen manifests -- never changed here.

    python3 phase4_draws.py            # writes <private>/paper/phase4/draws/<mod>/...json
"""
from __future__ import annotations
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json
import os
import numpy as np

ROOT = _P
OUT = f"{ROOT}/paper/phase4/draws"
NS = [3, 5, 10, 15, 20, 30, 40]
SEEDS = [42, 1, 7]
CLASS = lambda n: 0 if n == 0 else 1 if n <= 2 else 2 if n <= 5 else 3


def order_strat(cases, seed):
    """Proportional round-robin: take one scan of the largest remaining class per step."""
    rng = np.random.RandomState(seed)
    by_class = {c: list(ids) for c, ids in
                ((c, [e["id"] for e in cases if CLASS(e["n_cmb"]) == c]) for c in (0, 1, 2, 3))}
    for c in by_class:
        rng.shuffle(by_class[c])
    order, remaining = [], {c: list(v) for c, v in by_class.items()}
    total = sum(len(v) for v in remaining.values())
    quota = {c: len(v) / total for c, v in remaining.items()}
    drawn = {c: 0 for c in quota}
    while sum(len(v) for v in remaining.values()):
        # take from the class furthest behind its proportional quota (round-robin by share)
        c = min((c for c in remaining if remaining[c]),
                key=lambda c: (drawn[c] - quota[c] * len(order), c))
        order.append(remaining[c].pop())
        drawn[c] += 1
    return order


def order_rand(cases, seed):
    rng = np.random.RandomState(seed)
    ids = [e["id"] for e in cases]
    rng.shuffle(ids)
    return ids


def order_low(cases, seed):
    """31.10.2026 (user: "high, random and low"): the LIGHT cases first -- scans with at least one CMB, fewest CMB first
    (cheapest to annotate); CMB-free scans only after all scans with CMB (otherwise small n would contain no lesion)."""
    n = {e["id"]: e["n_cmb"] for e in cases}
    return sorted(n, key=lambda i: (n[i] == 0, n[i], i))


def order_mid(cases, seed):
    """31.10.2026 (user: "no, low, middle and high"): scans with >= 1 CMB closest to the MEDIAN count of those scans first;
    CMB-free scans last."""
    n = {e["id"]: e["n_cmb"] for e in cases}
    med = float(np.median([v for v in n.values() if v > 0]))
    return sorted(n, key=lambda i: (n[i] == 0, abs(n[i] - med), i))


def order_none(cases, seed):
    """31.10.2026 (user): ONLY CMB-free scans (random order per seed) -- negative-only fine-tuning; n is capped at the
    number of CMB-free training scans (the draw file says capped)."""
    rng = np.random.RandomState(seed)
    ids = [e["id"] for e in cases if e["n_cmb"] == 0]
    rng.shuffle(ids)
    return ids


def order_high(cases, seed):
    return sorted((e["id"] for e in cases), key=lambda i: (-next(e["n_cmb"] for e in cases if e["id"] == i), i))


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for mod, manifest in (("t2star", f"{ROOT}/gitter/manifest_t2star.json"),
                          ("swi", f"{ROOT}/gitter/manifest_swi.json")):
        entries = json.load(open(manifest))
        dmod = os.path.join(OUT, mod)
        os.makedirs(dmod, exist_ok=True)
        summary[mod] = {}
        for k in range(5):
            train = [e for e in entries if e["fold"] != k]
            for strategy in ("STRAT", "RAND", "HIGH", "LOW", "MID", "NONE"):      # 31.10.2026: SWI too, as the wk0b spec says
                for seed in SEEDS:
                    fn = {"STRAT": order_strat, "RAND": order_rand, "HIGH": order_high, "LOW": order_low, "MID": order_mid, "NONE": order_none}[strategy]
                    order = fn(train, seed)
                    for n in NS:
                        take = n if n < len(order) else len(order)        # NONE: order holds only the CMB-free scans
                        taken = order[:take]
                        n_cmb = sum(e["n_cmb"] for e in train if e["id"] in taken)
                        path = f"{dmod}/fold{k}_n{n}_{strategy}_{seed}.json"
                        json.dump(dict(fold=k, n=n, n_actual=take, capped=take < n, strategy=strategy, seed=seed,
                                       cases=taken, n_cmb=n_cmb, has_cmb=n_cmb > 0,
                                       source_manifest=manifest), open(path, "w"), indent=1)
                    # RAND report: how often small draws contain no CMB scan at all
                    if strategy == "RAND":
                        empty = [n for n in NS[:3] if sum(e["n_cmb"] for e in train if e["id"] in order[:n]) == 0]
                        summary[mod][f"fold{k}_{strategy}_seed{seed}"] = dict(no_cmb_draws=empty)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    n_files = sum(len([f for f in os.listdir(os.path.join(OUT, m)) if f.endswith(".json")])
                  for m in ("t2star", "swi"))
    print(f"wrote {n_files} draw files under {OUT}")


if __name__ == "__main__":
    main()
