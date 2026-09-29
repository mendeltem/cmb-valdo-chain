"""Summarise the verdicts written by the review page (open or blinded).

    python -m cmb.review.summarise_review VERDICTS.json [--key <out>.key.json] [--json OUT.json]

What it reports
---------------
1. Per kind of candidate (false positive / missed / detected) how the expert judged them, and for
   "not a microbleed" the reasons.
2. With the key of a blinded page: the scores if the expert's verdicts REPLACED the reference
   annotation -- symmetrically, so the adjudication cannot only help the network:

   * a "false positive" judged to be a microbleed becomes a true positive (the annotators overlooked it);
   * a "missed" reference lesion judged not to be a microbleed stops counting as a miss;
   * a "detected" reference lesion judged not to be a microbleed is removed from the reference, and the
     detection that stood for it becomes a false positive.

   "unsure" and candidates without a verdict keep their original status. A reference lesion that is shown
   by several candidates is only removed if every judged one says "not a microbleed".
   Pooled lesion-level F1 with a bootstrap over cases (2000 resamples) for the adjudicated value and for
   the paired difference to the original.
3. Intra-rater agreement on the candidates that were shown twice (raw agreement and Cohen's kappa).
   For the adjudication the FIRST viewing counts.

Limits to state with the numbers: one expert; only candidates of the network or the reference were
shown, so lesions that both overlooked stay invisible (the adjudicated sensitivity is an upper bound).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

KINDS = ("false_positive", "missed", "detected")
VERDICTS = ("microbleed", "not_microbleed", "unsure")


def f1_of(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """-> (F1, precision, sensitivity); nan where undefined."""
    precision = tp / (tp + fp) if tp + fp else float("nan")
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return f1, precision, sensitivity


def case_counts(candidates: List[Dict], verdict_of: Dict[str, Optional[str]], adjudicated: bool) -> Dict[str, Tuple[int, int, int]]:
    """-> {case: (tp, fp, fn)} from the primary candidates (repeats excluded), original or adjudicated."""
    by_reference: Dict[str, List[Optional[str]]] = defaultdict(list)          # reference lesion -> verdicts of the candidates showing it
    for c in candidates:
        for r in c["reference_ids"]:
            by_reference[r].append(verdict_of.get(c["id"]))

    def rejected(reference_id: str) -> bool:
        judged = [v for v in by_reference[reference_id] if v in ("microbleed", "not_microbleed")]
        return adjudicated and bool(judged) and all(v == "not_microbleed" for v in judged)

    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    detected_refs, missed_refs = set(), set()
    for c in candidates:
        case, verdict = c["case"], verdict_of.get(c["id"])
        if c["kind"] == "false_positive":
            if adjudicated and verdict == "microbleed":
                counts[case][0] += 1                                          # overlooked by the annotators
            else:
                counts[case][1] += 1
        elif c["kind"] == "detected":
            detected_refs.update(c["reference_ids"])
            if all(rejected(r) for r in c["reference_ids"]):
                counts[case][1] += 1                                          # it stood for lesions the expert does not accept
        else:
            missed_refs.update(c["reference_ids"])
    for r in detected_refs:
        if not rejected(r):
            counts[r.split("#")[0]][0] += 1
    for r in missed_refs - detected_refs:
        if not rejected(r):
            counts[r.split("#")[0]][2] += 1
    return {case: tuple(v) for case, v in counts.items()}


def pooled(counts: Dict[str, Tuple[int, int, int]], cases: List[str]) -> Tuple[int, int, int]:
    total = np.zeros(3, int)
    for case in cases:
        total += np.array(counts.get(case, (0, 0, 0)))
    return int(total[0]), int(total[1]), int(total[2])


def kappa(pairs: List[Tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    observed = sum(a == b for a, b in pairs) / len(pairs)
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[v] * second[v] for v in VERDICTS) / len(pairs) ** 2
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("verdicts", help="review_<...>.json written by the page")
    parser.add_argument("--key", help="<out>.key.json of a blinded page (needed for blinded verdict files)")
    parser.add_argument("--json", help="write the summary here as well")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    review = json.load(open(args.verdicts))
    judged = {r["id"]: r for r in review["candidates"]}
    print(f"{review['dataset']} / {review['model']}: {len(judged)} judged candidates (saved {review['saved']})")
    key = json.load(open(args.key)) if args.key else None
    if key is None and any("kind" not in r for r in judged.values()):
        raise SystemExit("this is a blinded verdict file: give --key <out>.key.json")

    rows = key["candidates"] if key else list(judged.values())
    out: Dict = dict(dataset=review["dataset"], model=review["model"], judged=len(judged), kinds={})
    primary = [c for c in rows if c.get("repeat_of") is None]
    # The first viewing counts (by time stamp): of a candidate shown twice, the later verdict only feeds the agreement.
    twin = {c["repeat_of"]: c["id"] for c in rows if c.get("repeat_of")}

    def first_record(candidate_id: str) -> Optional[Dict]:
        seen = [judged[i] for i in (candidate_id, twin.get(candidate_id)) if i in judged]
        return min(seen, key=lambda r: r.get("time", "")) if seen else None

    records = {c["id"]: first_record(c["id"]) for c in primary}
    for kind in KINDS:
        of_kind = [records[c["id"]] for c in primary if c["kind"] == kind]
        verdicts = Counter(r["verdict"] for r in of_kind if r)
        reasons = Counter(r.get("reason") or "none given" for r in of_kind if r and r["verdict"] == "not_microbleed")
        out["kinds"][kind] = dict(shown=len(of_kind), judged=sum(verdicts.values()), verdicts=dict(verdicts), reasons=dict(reasons))
        print(f"  {kind:15s} shown {len(of_kind):3d}  judged {sum(verdicts.values()):3d}  {dict(verdicts)}   reasons for 'not a microbleed': {dict(reasons)}")

    if key is None:
        overlooked = sum(r["kind"] == "false_positive" and r["verdict"] == "microbleed" for r in judged.values())
        doubtful = sum(r["kind"] == "missed" and r["verdict"] == "not_microbleed" for r in judged.values())
        print(f"  -> {overlooked} 'false positives' are microbleeds the annotators overlooked; {doubtful} 'missed' reference lesions are not "
              f"microbleeds according to the expert.  (Adjudicated scores need a blinded page and its key.)")
    else:
        verdict_of = {candidate_id: (record["verdict"] if record else None) for candidate_id, record in records.items()}
        cases = key.get("cases") or sorted({c["case"] for c in primary})
        original = case_counts(primary, verdict_of, adjudicated=False)
        adjudicated = case_counts(primary, verdict_of, adjudicated=True)
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(cases), size=(args.resamples, len(cases)))
        boot = np.array([[f1_of(*pooled(counts, [cases[i] for i in draw]))[0] for counts in (original, adjudicated)] for draw in draws])
        for name, counts, column in (("reference as delivered", original, 0), ("expert-adjudicated", adjudicated, 1)):
            tp, fp, fn = pooled(counts, cases)
            f1, precision, sensitivity = f1_of(tp, fp, fn)
            lo, hi = np.nanpercentile(boot[:, column], [2.5, 97.5])
            out[name] = dict(tp=tp, fp=fp, fn=fn, f1=round(f1, 3), precision=round(precision, 3), sensitivity=round(sensitivity, 3),
                             f1_ci=[round(float(lo), 3), round(float(hi), 3)])
            print(f"  {name:24s} TP {tp:3d}  FP {fp:3d}  FN {fn:3d}   F1 {f1:.3f} [{lo:.3f}; {hi:.3f}]  P {precision:.2f}  S {sensitivity:.2f}")
        delta = boot[:, 1] - boot[:, 0]
        lo, hi = np.nanpercentile(delta, [2.5, 97.5])
        point = out["expert-adjudicated"]["f1"] - out["reference as delivered"]["f1"]
        out["paired_difference"] = dict(delta=round(point, 3), ci=[round(float(lo), 3), round(float(hi), 3)], cases=len(cases))
        print(f"  paired  adjudicated - delivered  [n={len(cases)} cases]:  {point:+.3f}  [{lo:+.3f}; {hi:+.3f}]")

        pairs = [(judged[c["repeat_of"]]["verdict"], judged[c["id"]]["verdict"]) for c in rows
                 if c.get("repeat_of") and c["id"] in judged and c["repeat_of"] in judged]
        if pairs:
            agreement = sum(a == b for a, b in pairs) / len(pairs)
            out["intra_rater"] = dict(pairs=len(pairs), agreement=round(agreement, 3), kappa=round(kappa(pairs), 3))
            print(f"  intra-rater: {len(pairs)} candidates judged twice, agreement {agreement:.2f}, Cohen's kappa {kappa(pairs):.2f}")
        else:
            print("  intra-rater: no candidate judged twice yet")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
