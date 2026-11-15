"""Decide the Arena-2 stage-2 arms (TrackSet) by rules fixed BEFORE the runs (v2, 2026-10-25).

Input: JSON files of cmb.stage2.pooled (any mix of arms and --seed-offset values; all on the SAME candidate sets).

Why v2 replaced v1 the same day, before any arm had run (literature + simulation check, scratchpad round3/R2_decision_stats):
v1 drew a separate bootstrap per arm and cohort, left the seed variance out of the interval, used '2 x the control's seed sd' (2 df,
a random threshold) and had no multiplicity adjustment; with 10 arms against one control its simulated family-wise rate of a false
KEEP was 17-23 %. v2:
  * ONE case bootstrap per draw, shared by all arms and seeds and stratified by cohort, so the 10 comparisons keep their joint
    structure (one control, one case set);
  * interval = case part + seed part: se_total = sqrt(se_case^2 + 2 sigma_seed^2 / S), sigma_seed pooled over all arms that have
    >= 2 seeds, never below the measured stage-2 seed sd 0.009 (VALDO, 2026-09-28);
  * max-T (Dunnett-type) simultaneous lower bound over the arms at a one-sided family-wise 10 % (selection level; the survivor is
    then tested ONCE on cases not used here);
  * a fixed practical margin: delta on 'all' >= +0.01;
  * shrunk delta (normal prior, sd 0.02) next to the raw one -- the raw delta of the best arm is inflated (winner's curse);
  * sensitivity: the delta on 'all' without the scan with the most reference lesions of each cohort.

Rules:
  screening (1 seed):   'advance to seed 1' if delta on 'all' >= +0.01 and no cohort <= -0.02, else 'stop (screening)'.
  selection (>= 2 seeds): 'carry to confirmation' if simultaneous lower bound > 0, delta on 'all' >= +0.01 and no cohort whose
                        95 % interval lies entirely below 0; else 'drop'. A third seed only helps when se_case < 0.005 (then the
                        line says so). Nothing selected here is called 'better' before the confirmation on fresh cases.
  d4 arms count only if they also carry against tset-tta8 and tset-wide-tta8 (run with --control for each).

    python -m cmb.stage2.tset_summary --json $MB/stage2/tset_*.json --control "pooled tset-plain" [--out summary.json]
Prints sums only -- safe for the private cohorts.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

SIGMA_SEED_FLOOR = 0.009
TAU = 0.02
DELTA_MIN = 0.01
FAMILY_LEVEL = 0.10


def f1_arr(a: np.ndarray) -> float:
    tp, fp, fn = a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def f1(triples) -> float:
    return f1_arr(np.asarray(list(triples), float).reshape(-1, 3))


def load(files: List[str], tags: Dict[str, str] = None) -> Dict[Tuple[int, str], Dict[str, Dict[str, tuple]]]:
    """(seed_offset, mode) -> cohort -> case -> (tp, fp, fn); later files win on duplicates.
    tags: path -> tag; every mode of that file becomes '<mode> <tag>' -- for runs on a DIFFERENT candidate set of the SAME cases
    (e.g. the union of several stage-1 runs, wc7), whose mode names would otherwise collide with the base runs."""
    runs = {}
    tags = {os.path.abspath(k): v for k, v in (tags or {}).items()}
    for path in files:
        data = json.load(open(path))
        seed = int(data.get("seed_offset", 0))
        tag = tags.get(os.path.abspath(path))
        for mode, per_cohort in data["per_case"].items():
            runs[(seed, f"{mode} {tag}" if tag else mode)] = {c: {k: tuple(v) for k, v in cases.items()} for c, cases in per_cohort.items()}
    return runs


def summarise(runs, control: str, draws: int = 4000, seed: int = 0) -> Dict[str, Dict]:
    rng = np.random.RandomState(seed)
    seeds_of: Dict[str, List[int]] = {}
    for s, m in runs:
        seeds_of.setdefault(m, []).append(s)
    for m in seeds_of:
        seeds_of[m] = sorted(seeds_of[m])
    ref = runs[(seeds_of[control][0], control)]
    cohorts = sorted(ref)
    cases = [(c, k) for c in cohorts for k in sorted(ref[c])]
    coh = np.array([c for c, _ in cases])
    strata = [np.nonzero(coh == c)[0] for c in cohorts]

    def arr(s, m):
        per = runs[(s, m)]
        return np.array([per[c][k] for c, k in cases], float)

    # pooled seed sd of F1 on 'all' over every mode with >= 2 seeds (control included)
    var, df = 0.0, 0
    for m, ss in seeds_of.items():
        if len(ss) >= 2:
            v = [f1_arr(arr(s, m)) for s in ss]
            var += np.var(v, ddof=1) * (len(v) - 1); df += len(v) - 1
    sigma_seed = max(float(np.sqrt(var / df)) if df else 0.0, SIGMA_SEED_FLOOR)

    arms = [m for m in sorted(seeds_of) if m != control and set(seeds_of[m]) & set(seeds_of[control])]
    data, hat, hat_c = {}, {}, {}
    for m in arms:
        ss = sorted(set(seeds_of[m]) & set(seeds_of[control]))
        A = [arr(s, m) for s in ss]; C = [arr(s, control) for s in ss]
        data[m] = (A, C, ss)
        hat[m] = float(np.mean([f1_arr(a) - f1_arr(c) for a, c in zip(A, C)]))
        hat_c[m] = {coh_name: float(np.mean([f1_arr(a[idx]) - f1_arr(c[idx]) for a, c in zip(A, C)]))
                    for coh_name, idx in zip(cohorts, strata)}
    boot = {m: np.zeros(draws) for m in arms}
    boot_c = {m: {c: np.zeros(draws) for c in cohorts} for m in arms}
    for d in range(draws):
        picks = [rng.choice(idx, size=len(idx)) for idx in strata]
        ix = np.concatenate(picks)
        for m in arms:
            A, C, _ = data[m]
            boot[m][d] = np.mean([f1_arr(a[ix]) - f1_arr(c[ix]) for a, c in zip(A, C)])
            for c_name, p in zip(cohorts, picks):
                boot_c[m][c_name][d] = np.mean([f1_arr(a[p]) - f1_arr(c[p]) for a, c in zip(A, C)])
    se_case = {m: float(np.std(boot[m], ddof=1)) or 1e-9 for m in arms}
    crit = float(np.quantile(np.max([(boot[m] - hat[m]) / se_case[m] for m in arms], axis=0), 1 - FAMILY_LEVEL)) if arms else 0.0

    heavy = {c: max(ref[c], key=lambda k: ref[c][k][0] + ref[c][k][2]) for c in cohorts}     # most reference lesions (tp + fn)
    keep_light = np.array([k != heavy[c] for c, k in cases])
    result = {}
    for m in arms:
        A, C, ss = data[m]
        S = len(ss)
        se_total = float(np.sqrt(se_case[m] ** 2 + 2 * sigma_seed ** 2 / S))
        lower = hat[m] - crit * se_total
        cohort_ci = {c: [round(float(np.percentile(boot_c[m][c], 2.5)), 4), round(float(np.percentile(boot_c[m][c], 97.5)), 4)] for c in cohorts}
        light = float(np.mean([f1_arr(a[keep_light]) - f1_arr(c[keep_light]) for a, c in zip(A, C)]))
        e = dict(seeds=ss, delta_all=round(hat[m], 4), delta_cohorts={c: round(v, 4) for c, v in hat_c[m].items()},
                 ci95_cohorts=cohort_ci, se_case=round(se_case[m], 4), sigma_seed=round(sigma_seed, 4), se_total=round(se_total, 4),
                 crit_maxT=round(crit, 3), simultaneous_lower=round(lower, 4),
                 shrunk_delta=round(hat[m] * TAU ** 2 / (TAU ** 2 + se_total ** 2), 4), delta_all_without_heaviest_scans=round(light, 4))
        e["decision"] = decide(e)
        result[m] = e
    return result


def decide(e: Dict) -> str:
    if len(e["seeds"]) < 2:
        ok = e["delta_all"] >= DELTA_MIN and not any(v <= -0.02 for v in e["delta_cohorts"].values())
        return "advance to seed 1" if ok else "stop (screening)"
    harm = any(hi < 0 for _, hi in e["ci95_cohorts"].values())
    carry = e["simultaneous_lower"] > 0 and e["delta_all"] >= DELTA_MIN and not harm
    if carry:
        return "carry to confirmation"
    if e["se_case"] < 0.005 and e["delta_all"] >= DELTA_MIN and len(e["seeds"]) == 2 and not harm:
        return "drop (a third seed could still decide: se_case < 0.005)"
    return "drop"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", nargs="+", required=True)
    parser.add_argument("--control", default="pooled tset-plain")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--tag", nargs="*", default=[], metavar="FILE=TAG",
                        help="modes of FILE get the suffix TAG (runs on another candidate set of the same cases)")
    parser.add_argument("--out")
    args = parser.parse_args()
    tags = dict(s.split("=", 1) for s in args.tag)
    res = summarise(load(args.json + [f for f in tags if f not in args.json], tags), args.control, draws=args.draws)
    for mode, e in res.items():
        print(f"{mode:28s} seeds {e['seeds']}  all {e['delta_all']:+.3f} (shrunk {e['shrunk_delta']:+.3f}, lower {e['simultaneous_lower']:+.3f}, "
              f"se case {e['se_case']:.3f} / total {e['se_total']:.3f})  "
              + "  ".join(f"{c} {v:+.3f}" for c, v in e["delta_cohorts"].items()) + f"   -> {e['decision']}")
    if args.out:
        json.dump(dict(control=args.control, rules="v2 2026-10-25", results=res), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
