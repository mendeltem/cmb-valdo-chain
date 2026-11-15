"""Phase-4 stage-2 chains, variant frozen-V: the VALDO-only tset-plain classifier, unchanged,
applied to each phase-4 block's target candidate set (the frozen test fold of D).

For every block folder candidates/<mod>/<V>_n<n>_fold<k>_s<seed>/ (that has candidates.json)
this scores:
  frozen-V (a)  threshold = the pooled pool-OOF F1 threshold of the VALDO-only classifier
  frozen-V (b)  threshold = the per-cohort (<label>-<mod>) pool-OOF F1 threshold of the
                same classifier  -- this is the 'calibrated on D's scans' variant only in the
                sense that the pool contains D's own folds; the classifier itself is unchanged.
The classifier is retrained here on the VALDO pool ONLY (same recipe as the wc4 final:
tset-plain, seed offset 0, 40 epochs, 3 members) and then applied to the target. It is never
trained on the target candidates (that is the 'updated' variant, a different script).

The pooled pool OOF threshold comes from re-running the pool OOF selection on the VALDO
candidates (cmb.stage2.tset_apply.run_arm with target = a throwaway empty-safe set is not
used: run_arm's pool-OOF selection is independent of the target, we just retrain).

Usage:  python chain_frozenV.py [--mod t2star] [--seed-offset 0] [--out CHAINS.json]
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
import sys
import glob

import numpy as np

ROOT = _R
MB = _P
P4 = f"{MB}/paper/phase4"
sys.path.insert(0, ROOT)
from cmb.stage2 import tset_apply as TA          # noqa: E402
from cmb.stage2 import trackset as T             # noqa: E402
from cmb.stage2.pooled import scores_of          # noqa: E402
from cmb.stage2.classifier import THRESHOLDS     # noqa: E402

VALDO_POOL = f"{ROOT}/ergebnisse/stage2/a03-aniso-e60-gd-weich-z40"
VALDO_FOLDS = f"{ROOT}/dev/manifest_valdo_gitter_d.json"


def valdo_only_classifier(seed_offset: int, epochs: int = 40, members: int = 3):
    """Rebuild the VALDO-only tset-plain classifier exactly as in the Arena runs, and return
    (rows, bundle, cases, pooled_threshold, per_cohort_threshold)."""
    rows, blocks, cases = TA.load_sets([f"valdo={VALDO_POOL}"], {})
    y = np.array([1 if r["touched"] else 0 for r in rows])
    fold = np.array([r["fold"] for r in rows])
    mods = T.ARMS["tset-plain"]
    fit = lambda tx, ty, sx, seed: T.fit_predict_trackset(tx, ty, sx, seed=seed, modules=mods,
                                                         epochs=epochs, members=members)
    # 04.11.2026: fit_predict_trackset needs a Bundle (it reads .patches); the bundle was built only after this loop
    bundle = TA.bundle_of(rows, blocks, "track" in mods)
    oof = np.zeros(len(rows), np.float32)
    for k in sorted(set(fold)):
        held = fold == k
        oof[held] = fit(bundle[np.nonzero(~held)[0]], y[~held], bundle[np.nonzero(held)[0]],
                        1000 * seed_offset + k)
    n_ref = sum(c["n_reference"] for c in cases.values()); n_cases = len(cases)
    thr = float(max(THRESHOLDS, key=lambda t: scores_of(rows, oof >= t, n_ref, n_cases)["f1"]))
    per_cohort = {}
    for name in sorted({c["cohort"] for c in cases.values()}):
        sel = np.array([r["cohort"] == name for r in rows])
        here = [c for c in cases.values() if c["cohort"] == name]
        per_cohort[name] = float(max(THRESHOLDS, key=lambda t: scores_of(
            [r for r, s in zip(rows, sel) if s], oof[sel] >= t, sum(c["n_reference"] for c in here), len(here))["f1"]))   # 04.11.: was oof[sel] (scores, not decisions)
    return rows, bundle, cases, thr, per_cohort, oof


def concat_bundles(bundles):
    """One test Bundle from several target Bundles (case indices offset so they stay distinct)."""
    off, ci = 0, []
    for b in bundles:
        ci.append(b.case_index + off); off += int(b.case_index.max()) + 1 if len(b) else 0
    return T.Bundle(np.concatenate([np.asarray(b.patches) for b in bundles]), np.concatenate([b.side for b in bundles]),
                    np.concatenate(ci), np.concatenate([b.centre_mm for b in bundles]), None)


def score_group(target_dirs, seed_offset: int) -> dict:
    """04.11.2026: frozen-V = the SAME VALDO-only classifier for every target. Train it once per group of
    targets (one sequence x fold, 78 blocks) and apply it to all of them together, instead of retraining it for each
    of the 780 targets (deterministic training, fixed seed: the same model either way)."""
    mods = T.ARMS["tset-plain"]
    loaded, bundles, out = [], [], {}
    for d in target_dirs:
        if not os.path.exists(f"{d}/patches.npy"):
            # 04.11.: stage 1 proposed nothing (e.g. the CMB-free-only NONE draws): no patches.npy. The chain then finds 0 CMB:
            # all reference lesions are misses, F1 = 0 -- a result, not an error.
            info = json.load(open(f"{d}/candidates.json")); cs = info["cases"]
            n_ref = sum(c["n_reference"] for c in cs.values())
            res = dict(n_candidates=len(info.get("candidates", [])), empty=True)
            for label in ("a_pooled", "b_percohort"):
                res[label] = dict(threshold=VALDO_THR, **scores_of([], np.zeros(0, bool), n_ref, len(cs)), per_case={})
            out[d] = res; continue
        rows, blocks, cases = TA.load_sets([f"target={d}"], {})
        if not rows:
            out[d] = dict(empty=True, f1=float("nan"), n_candidates=0); continue
        loaded.append((d, rows, cases)); bundles.append(TA.bundle_of(rows, blocks, "track" in mods))
    if not loaded:
        return out
    y = np.array([1 if r["touched"] else 0 for r in VALDO_ROWS])
    scores_all = T.fit_predict_trackset(VALDO_BUNDLE, y, concat_bundles(bundles), seed=1000 * seed_offset + 99,
                                        modules=mods, epochs=40, members=3)
    pos = 0
    for (d, rows, cases), b in zip(loaded, bundles):
        scores = scores_all[pos:pos + len(b)]; pos += len(b)
        n_ref = sum(c["n_reference"] for c in cases.values()); n_cases = len(cases)
        res = dict(n_candidates=len(rows))
        for label, t in (("a_pooled", VALDO_THR),
                         ("b_percohort", VALDO_PER_COHORT.get(cases[sorted(cases)[0]]["cohort"], VALDO_THR))):
            acc = scores >= t
            res[label] = dict(threshold=t, **scores_of(rows, acc, n_ref, n_cases), per_case=TA.per_case(rows, acc, cases))
        out[d] = res
    return out


# module-level VALDO classifier, built once
VALDO_ROWS = VALDO_BUNDLE = VALDO_THR = VALDO_PER_COHORT = VALDO_OOF = None


def build_valdo(seed_offset: int):
    global VALDO_ROWS, VALDO_BUNDLE, VALDO_THR, VALDO_PER_COHORT, VALDO_OOF
    VALDO_ROWS, VALDO_BUNDLE, _, VALDO_THR, VALDO_PER_COHORT, VALDO_OOF = \
        valdo_only_classifier(seed_offset)
    print(f"VALDO-only tset-plain ready: pooled threshold {VALDO_THR}, "
          f"per-cohort {VALDO_PER_COHORT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", choices=["t2star", "swi"], default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", default=f"{P4}/chains_frozenV.json")
    a = ap.parse_args()
    build_valdo(a.seed_offset)
    out = {}
    cands = sorted(glob.glob(f"{P4}/candidates/*/*_fold*_s*"))   # 04.11.: was */*/* (one level too deep -> 0 targets)
    if a.mod:
        cands = [c for c in cands if f"/candidates/{a.mod}/" in c]
    import re, collections
    groups = collections.OrderedDict()
    for c in cands:
        if os.path.exists(f"{c}/candidates.json"):
            m = re.search(r"/candidates/([^/]+)/[^/]*_fold(\d)_", c)
            groups.setdefault((m.group(1), m.group(2)), []).append(c)
    for (mod, k), dirs in groups.items():
        try:
            res = score_group(dirs, a.seed_offset)
        except Exception as e:      # keep going; record the error, it is not a result
            res = {d: dict(error=f"{type(e).__name__}: {e}") for d in dirs}
            print(f"{mod} fold {k}: ERROR {type(e).__name__}: {e}", flush=True)
        for d, r in res.items():
            name = d.replace(f"{P4}/candidates/", ""); out[name] = r
        f1 = [r["a_pooled"]["f1"] for r in res.values() if "a_pooled" in r]
        print(f"{mod} fold {k}: {len(dirs)} blocks, mean frozen-V(a) F1 {np.mean(f1) if f1 else float('nan'):.3f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)          # partial results survive a crash
    json.dump(out, open(a.out, "w"), indent=1)
    print("written to", a.out)


if __name__ == "__main__":
    main()
