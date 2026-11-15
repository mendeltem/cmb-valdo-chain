"""Phase-4 stage-2 chains, variant frozen-A1v2: the VALDO-ONLY winner of the Arena 1 v2
selection (MB/stage2/a1v2/selection.json), unchanged, applied to each phase-4 block's
target candidate set (the frozen test fold of D).

Identical to chain_frozenV.py (which stays bit-identical) except the classifier: instead of
the hardcoded tset-plain it reads the winner arm from selection.json and rebuilds that arm
on the VALDO pool only (same recipe as cmb.stage2.pooled: same features, same fit,
epochs/members as the selection run). It is never trained on the target candidates.

Variants, same as frozen-V:
  (a) threshold = the pooled pool-OOF F1 threshold of the VALDO-only classifier
  (b) threshold = the per-cohort (<label>-<mod>) pool-OOF F1 threshold of the same
      classifier (calibrated on D's scans' thresholds only; classifier unchanged).

Usage:  python chain_frozenA1v2.py [--mod t2star] [--seed-offset 0] [--out CHAINS.json]
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
import time

import numpy as np

ROOT = _R
MB = _P
P4 = f"{MB}/paper/phase4"
sys.path.insert(0, ROOT)
from cmb.stage2 import tset_apply as TA          # noqa: E402
from cmb.stage2 import trackset as T             # noqa: E402
from cmb.stage2 import pooled as PL              # noqa: E402
from cmb.stage2.classifier import THRESHOLDS, features_of  # noqa: E402
from cmb.stage2.pooled import scores_of          # noqa: E402

VALDO_POOL = f"{ROOT}/ergebnisse/stage2/a03-aniso-e60-gd-weich-z40"
SELECTION = f"{MB}/stage2/a1v2/selection.json"
EPOCHS = 40
MEMBERS = 3


def winner_arm():
    s = json.load(open(SELECTION))
    arm = s["winner"]
    assert arm in ("tset-plain", "feat-vessel", "feat-vessel-tissue"), \
        f"winner arm '{arm}' not implementable here (expected one of the 5 final arms)"
    return arm


def build_inputs(arm: str, rows, patches_all, tissue):
    """Feature table for the arm, exactly as cmb.stage2.pooled builds it (base features +
    the arm's extras). patches_all is a full (not mmap) array."""
    x = np.asarray([features_of(np.asarray(patches_all[i]), r, THRESHOLDS[0]) for i, r in enumerate(rows)],
                   np.float32)
    if arm == "tset-plain":
        return None, None
    case_ids = {c: i for i, c in enumerate(sorted({r["case"] for r in rows}))}
    case_index = np.array([case_ids[r["case"]] for r in rows])
    centre_mm = np.array([r["centre_voxel"] for r in rows], np.float32) * T.VOXEL_MM_XYZ
    neighbours = T.neighbour_features(case_index, centre_mm, x[:, 0])
    if arm == "feat-vessel":
        return np.concatenate([x, neighbours], 1), None
    if arm == "feat-vessel-tissue":
        return np.concatenate([x, neighbours, tissue], 1), None
    raise AssertionError(arm)


def valdo_only_classifier(arm: str, seed_offset: int, epochs: int = EPOCHS, members: int = MEMBERS):
    """Rebuild the VALDO-only classifier of the given arm and return
    (rows, fit_predict, inputs, pooled_threshold, per_cohort_threshold)."""
    rows, blocks, cases = TA.load_sets([f"valdo={VALDO_POOL}"], {})
    y = np.array([1 if r["touched"] else 0 for r in rows])
    fold = np.array([r["fold"] for r in rows])
    patches_all = np.load(f"{VALDO_POOL}/patches.npy")
    tissue = np.zeros((len(rows), 5), np.float32)
    for i, r in enumerate(rows):
        if r.get("tissue", -1) in PL.TISSUE_CLASS:
            tissue[i, PL.TISSUE_CLASS[r["tissue"]]] = 1.0
    if arm in PL.FEATURE_MODELS:
        kind, fit = PL.FEATURE_MODELS[arm]
        inputs, _ = build_inputs(arm, rows, patches_all, tissue)
    else:
        mods = T.ARMS["tset-plain"]
        bundle = TA.bundle_of(rows, blocks, "track" in mods)
        fit = lambda tx, ty, sx, seed: T.fit_predict_trackset(tx, ty, sx, seed=seed, modules=mods,
                                                              epochs=epochs, members=members)
        inputs = bundle
    oof = np.zeros(len(rows), np.float32)
    for k in sorted(set(fold)):
        held = fold == k
        oof[held] = fit(inputs[np.nonzero(~held)[0]], y[~held], inputs[np.nonzero(held)[0]],
                        1000 * seed_offset + k)
    global VALDO_ROWS, VALDO_INPUTS, VALDO_ARM
    VALDO_ROWS = rows; VALDO_INPUTS = inputs; VALDO_ARM = arm
    n_ref = sum(c["n_reference"] for c in cases.values()); n_cases = len(cases)
    thr = float(max(THRESHOLDS, key=lambda t: scores_of(rows, oof >= t, n_ref, n_cases)["f1"]))
    per_cohort = {}
    for name in sorted({c["cohort"] for c in cases.values()}):
        sel = np.array([r["cohort"] == name for r in rows])
        here = [c for c in cases.values() if c["cohort"] == name]
        per_cohort[name] = float(max(THRESHOLDS, key=lambda t: scores_of(
            [r for r, s in zip(rows, sel) if s], oof[sel] >= t, sum(c["n_reference"] for c in here), len(here))["f1"]))
    return rows, fit, inputs, thr, per_cohort


def concat_bundles(bundles):
    """One test Bundle from several target Bundles (case indices offset so they stay distinct)."""
    off, ci = 0, []
    for b in bundles:
        ci.append(b.case_index + off); off += int(b.case_index.max()) + 1 if len(b) else 0
    return T.Bundle(np.concatenate([np.asarray(b.patches) for b in bundles]), np.concatenate([b.side for b in bundles]),
                    np.concatenate(ci), np.concatenate([b.centre_mm for b in bundles]), None)


def score_group(target_dirs, seed_offset: int, arm: str, fit, inputs, thr_a, per_cohort,
                mod: str = "target", decisions_fh=None) -> dict:
    """04.11.2026 (kept from chain_frozenV.py): frozen-A1v2 = the SAME VALDO-only
    classifier for every target. Train it once per group (one modality x fold, 78 blocks)
    and apply it to all of them together."""
    mods = T.ARMS["tset-plain"] if arm == "tset-plain" else None
    loaded, bundles, inputs_list, out = [], [], [], {}
    for d in target_dirs:
        if not os.path.exists(f"{d}/patches.npy"):
            # 04.11.: stage 1 proposed nothing (e.g. the CMB-free-only NONE draws): no patches.npy.
            # The chain then finds 0 CMB: all reference lesions are misses, F1 = 0 -- a result, not an error.
            info = json.load(open(f"{d}/candidates.json")); cs = info["cases"]
            n_ref = sum(c["n_reference"] for c in cs.values())
            res = dict(n_candidates=len(info.get("candidates", [])), empty=True)
            for label in ("a_pooled", "b_percohort"):
                res[label] = dict(threshold=thr_a, **scores_of([], np.zeros(0, bool), n_ref, len(cs)), per_case={})
            out[d] = res; continue
        rows, blocks, cases = TA.load_sets([f"target={d}"], {})
        if not rows:
            out[d] = dict(empty=True, f1=float("nan"), n_candidates=0); continue
        if arm == "tset-plain":
            loaded.append((d, rows, cases)); bundles.append(TA.bundle_of(rows, blocks, "track" in mods))
        else:
            tissue = np.zeros((len(rows), 5), np.float32)
            for i, r in enumerate(rows):
                if r.get("tissue", -1) in PL.TISSUE_CLASS:
                    tissue[i, PL.TISSUE_CLASS[r["tissue"]]] = 1.0
            inputs_i, _ = build_inputs(arm, rows, np.load(f"{d}/patches.npy"), tissue)
            loaded.append((d, rows, cases)); inputs_list.append(inputs_i)
    if not loaded:
        return out
    if arm == "tset-plain":
        test_in = concat_bundles(bundles)
    else:
        test_in = np.concatenate(inputs_list, 0)
    scores_all = fit(inputs, np.array([1 if r["touched"] else 0 for r in VALDO_ROWS]), test_in,
                     seed=1000 * seed_offset + 99)
    pos = 0
    for (d, rows, cases), piece in zip(loaded, bundles if arm == "tset-plain" else inputs_list):
        n = len(piece)
        scores = scores_all[pos:pos + n]; pos += n
        n_ref = sum(c["n_reference"] for c in cases.values()); n_cases = len(cases)
        res = dict(n_candidates=len(rows))
        for label, t in (("a_pooled", thr_a),
                         ("b_percohort", per_cohort.get(cases[sorted(cases)[0]]["cohort"], thr_a))):
            acc = scores >= t
            res[label] = dict(threshold=t, **scores_of(rows, acc, n_ref, n_cases), per_case=TA.per_case(rows, acc, cases))
        if decisions_fh is not None:
            acc_a = scores >= thr_a
            acc_b = scores >= per_cohort.get(cases[sorted(cases)[0]]["cohort"], thr_a)
            for r, sc, aa, ab in zip(rows, scores, acc_a, acc_b):
                decisions_fh.write(json.dumps(dict(
                    cohort=mod, case=r["case"], fold=r["fold"], touched=list(r["touched"]),
                    probability=float(sc), accepted=bool(aa), probability_b=float(sc), accepted_b=bool(ab)
                )) + "\n")
        out[d] = res
    return out


# module-level VALDO classifier, built once
VALDO_ROWS = None
VALDO_FIT = None
VALDO_INPUTS = None
VALDO_THR = None
VALDO_PER_COHORT = None
VALDO_ARM = None


def build_valdo(seed_offset: int):
    global VALDO_ROWS, VALDO_FIT, VALDO_INPUTS, VALDO_THR, VALDO_PER_COHORT, VALDO_ARM
    VALDO_ARM = winner_arm()
    t0 = time.time()
    VALDO_ROWS, VALDO_FIT, VALDO_INPUTS, VALDO_THR, VALDO_PER_COHORT = \
        valdo_only_classifier(VALDO_ARM, seed_offset)
    print(f"VALDO-only {VALDO_ARM} ready in {time.time() - t0:.0f} s: pooled threshold {VALDO_THR}, "
          f"per-cohort {VALDO_PER_COHORT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", choices=["t2star", "swi"], default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", default=f"{P4}/chains_frozenA1v2.json")
    ap.add_argument("--dump-decisions", default=None,
                    help="JSONL: one line per candidate {cohort,case,fold,touched,probability,accepted,accepted_b}")
    a = ap.parse_args()
    build_valdo(a.seed_offset)
    out = {}
    cands = sorted(glob.glob(f"{P4}/candidates/*/*_fold*_s*"))
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
            res = score_group(dirs, a.seed_offset, VALDO_ARM, VALDO_FIT, VALDO_INPUTS, VALDO_THR, VALDO_PER_COHORT,
                              mod=mod, decisions_fh=open(a.dump_decisions, "a") if a.dump_decisions else None)
        except Exception as e:      # keep going; record the error, it is not a result
            res = {d: dict(error=f"{type(e).__name__}: {e}") for d in dirs}
            print(f"{mod} fold {k}: ERROR {type(e).__name__}: {e}", flush=True)
        for d, r in res.items():
            name = d.replace(f"{P4}/candidates/", ""); out[name] = r
        f1 = [r["a_pooled"]["f1"] for r in res.values() if "a_pooled" in r]
        print(f"{mod} fold {k}: {len(dirs)} blocks, mean frozen-A1v2({VALDO_ARM})(a) F1 "
              f"{np.mean(f1) if f1 else float('nan'):.3f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)          # partial results survive a crash
    json.dump(out, open(a.out, "w"), indent=1)
    print("written to", a.out)


if __name__ == "__main__":
    main()
