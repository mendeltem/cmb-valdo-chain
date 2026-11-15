"""wk11c, step 3 of 6: CMB recall by size class, with ONE diameter definition and
ONE-TO-ONE matching of accepted candidates to reference CMB.

CPU only, reads result files already on disk, writes no result file of the arena.

Inputs (read-only):
  MB/paper/phase4/a1v2_decisions_sizedump.jsonl  (streamed; mode "frozenA1v2(a)" only)
  MB/gitter/manifest_t2star.json, MB/gitter/manifest_swi_v2.json  (label files)

Diameter definition (the one of Table 2 and of 2.7): a reference CMB is a 26-connected
component of the reference mask; its diameter is the diameter of the circle with the area
of its largest cross-section on one slice (slice axis = largest voxel spacing), i.e.
2*sqrt(area/pi) with area = the largest in-plane area of the component in mm^2.
Classes: < 2 mm, 2-5 mm, > 5 mm.

Two recall semantics, per size class:
  chain_recall_any          = a reference CMB counts as found if ANY accepted candidate
                              touches it (the semantics used so far);
  chain_recall_one_to_one   = a reference CMB counts as found only if an accepted
                              candidate is matched to it, each accepted candidate
                              matching at most one reference CMB. Matching = maximum
                              cardinality bipartite matching per case, Hungarian
                              (scipy.optimize.linear_sum_assignment) on the 0/1
                              candidate-lesion matrix.

Repeated lines of the dump (the frozen-A1v2 chain is n-independent and was dumped once per
draw) are collapsed to unique candidate instances; a candidate instance is
(cohort, case, fold, tuple(touched), exact score).
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
from collections import defaultdict

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

MB = _P
P4 = f"{MB}/paper/phase4"
MANIFEST = {"t2star": f"{MB}/gitter/manifest_t2star.json", "swi": f"{MB}/gitter/manifest_swi_v2.json"}
CLASSES = (("<2 mm", 0.0, 2.0), ("2-5 mm", 2.0, 5.0), (">5 mm", 5.0, 1e9))
C26 = np.ones((3, 3, 3), bool)
DEFINITION = ("diameter = diameter of the circle with the area of the largest cross-section of the "
              "26-connected reference component on one slice (slice axis = largest voxel spacing); "
              "classes < 2 mm, 2-5 mm, > 5 mm")


def size_class(d):
    for name, lo, hi in CLASSES:
        if lo <= d < hi:
            return name
    return ">5 mm"


def diameters(path):
    """lesion id (1-based, 26-connected, same numbering as cmb.stage2.candidates) -> diameter in mm"""
    import nibabel as nib
    img = nib.load(path)
    zooms = img.header.get_zooms()[:3]
    ref, _ = ndimage.label(np.asarray(img.dataobj) > 0, structure=C26)
    ax = int(np.argmax(zooms))
    inplane = float(np.prod([z for i, z in enumerate(zooms) if i != ax]))
    out = {}
    for les, box in enumerate(ndimage.find_objects(ref), 1):
        m = ref[box] == les
        area = m.sum(axis=tuple(i for i in range(3) if i != ax)).max() * inplane
        out[les] = float(2 * np.sqrt(area / np.pi))
    return out


def read_dump(path, mode):
    """(cohort, case) -> (lesions touched by any candidate, set of unique accepted candidate
    instances, None). An accepted instance = (fold, tuple(touched), score)."""
    any_touched = defaultdict(set)
    accepted = defaultdict(set)
    for line in open(path):
        r = json.loads(line)
        if r["mode"] != mode:
            continue
        key = (r["cohort"], r["case"])
        any_touched[key].update(r["touched"])
        if r["accepted"]:
            accepted[key].add((r["fold"], tuple(sorted(r["touched"])), float(r["score"])))
    return any_touched, accepted


def match_one_to_one(touched_lists, n_les):
    """maximum cardinality matching of accepted candidates (list of touched-id lists) to
    lesions 1..n_les; returns the set of matched lesion ids."""
    rows = [t for t in touched_lists if t]
    if not rows or n_les == 0:
        return set()
    m = np.zeros((len(rows), n_les), np.uint8)
    for i, t in enumerate(rows):
        for j in t:
            if 1 <= j <= n_les:
                m[i, j - 1] = 1
    ri, ci = linear_sum_assignment(m, maximize=True)
    return {int(c) + 1 for r, c in zip(ri, ci) if m[r, c] == 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=f"{P4}/a1v2_decisions_sizedump.jsonl")
    ap.add_argument("--mode", default="frozenA1v2(a)")
    ap.add_argument("--out", default=_R + "/paper/wk11_sizeclasses.json")
    ap.add_argument("--limit", type=int, default=0, help="dry run: only the first N cases per sequence")
    a = ap.parse_args()

    any_touched, accepted = read_dump(a.dump, a.mode)

    res = {"definition": DEFINITION, "matching": "one-to-one, Hungarian maximum-cardinality "
           "matching per case (scipy.optimize.linear_sum_assignment)",
           "source_decisions": a.dump, "mode": a.mode, "cohorts": {}, "false_positives": {},
           "matching_effect": {},
           "note": ("per size class: 'chain_recall_any' (an accepted candidate touches the reference "
                    "CMB, the semantics used so far) and 'chain_recall_one_to_one' (each accepted "
                    "candidate matches at most one reference CMB). The dump pools every draw block of "
                    "the frozen chain (same classifier for every n), so the accepted-candidate counts "
                    "are pooled over those blocks and are NOT per-scan numbers.")}

    for seq, mf in MANIFEST.items():
        man = {e["id"]: e for e in json.load(open(mf))}
        cnt = defaultdict(int); ceil = defaultdict(int); tp_any = defaultdict(int); tp_one = defaultdict(int)
        fp_any = 0; fp_one = 0; n_cand_all = 0; n_cand_touching = 0; diam_all = []
        for case, e in sorted(man.items())[:a.limit or None]:
            key = (seq, case)
            dia = diameters(f"{e['dir']}/{case}_label.nii.gz")
            assert len(dia) == e["n_cmb"], f"{case}: {len(dia)} components vs manifest n_cmb {e['n_cmb']}"
            cls_of = {les: size_class(d) for les, d in dia.items()}
            diam_all.extend(dia.values())
            touches = [t for _, t, _ in sorted(accepted.get(key, ()))]
            touched_any = any_touched.get(key, set())
            matched = match_one_to_one(touches, len(dia))
            # false positives: accepted candidates matching no reference CMB (any) or not matched
            n_cand = len(touches)
            n_cand_all += n_cand
            n_cand_touching += sum(1 for t in touches if t)
            fp_any += sum(1 for t in touches if not t)
            fp_one += n_cand - len(matched)
            for les, cls in cls_of.items():
                for c in (cls, "all"):
                    cnt[c] += 1
                    ceil[c] += les in touched_any
                    tp_any[c] += les in touched_any and any(les in t for t in touches)
                    tp_one[c] += les in matched
        res["cohorts"][seq] = {
            cls: {"n": cnt[cls], "share": round(cnt[cls] / max(cnt["all"], 1), 3),
                  "stage1_ceiling_recall": round(ceil[cls] / max(cnt[cls], 1), 3),
                  "chain_recall_any": round(tp_any[cls] / max(cnt[cls], 1), 3),
                  "chain_recall_one_to_one": round(tp_one[cls] / max(cnt[cls], 1), 3),
                  "tp_any": tp_any[cls], "tp_one_to_one": tp_one[cls]}
            for cls in [c[0] for c in CLASSES] + ["all"]}
        res["cohorts"][seq]["median_diameter_mm"] = round(float(np.median(diam_all)), 2)
        res["matching_effect"][seq] = {
            "n_reference_cmb": cnt["all"],
            "tp_any": tp_any["all"], "tp_one_to_one": tp_one["all"],
            "reference_cmb_lost_by_one_to_one": tp_any["all"] - tp_one["all"],
            "accepted_candidates": n_cand_all,
            "accepted_candidates_touching_a_cmb": n_cand_touching,
            "accepted_candidates_not_matched_one_to_one": fp_one,
            "accepted_candidates_touching_nothing": fp_any}
        res["false_positives"][seq] = {"any_candidate_touches_nothing": fp_any,
                                       "one_to_one": fp_one,
                                       "n_accepted_candidates": n_cand_all,
                                       "n_accepted_candidates_touching_a_cmb": n_cand_touching}
    json.dump(res, open(a.out, "w"), indent=1)
    for seq in MANIFEST:
        c = res["cohorts"][seq]
        print(seq, {k: (v["n"], v["stage1_ceiling_recall"], v["chain_recall_any"],
                        v["chain_recall_one_to_one"]) for k, v in c.items() if isinstance(v, dict)})
        print("  matching effect", res["matching_effect"][seq])
    print("written to", a.out)


if __name__ == "__main__":
    main()
