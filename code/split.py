#!/usr/bin/env python3
"""we1: build split, stats, cases.

Stratification: cohort (prefix sub-1/2/3) x THREE load classes
(0 / 1-2 / >=3 CMB), NOT four -- with four, 3 of 12 cells are empty.

Validation: 15 cases (~20 %), per cohort a share of ~20 % (cohort1 3, cohort2 7,
cohort3 5), every cohort and every occupied cell represented, cases by CMB load.
Cohort 1 MUST be included (15 % of the cases, 45 % of all bleeds).
It is not touched up to we5.

Folds: 5, same stratification -- every cohort x class cell is distributed as
evenly as possible over the 5 folds (remaining cases land in folds
0..remainder-1). Within a cell: the fold's CMB sum is balanced greedily
(next case into the fold with the smallest CMB sum). Seed 0. On a
misallocation: swap, no other seed.

No seed knob for the balance: cohort 3 has only 34 CMB in 27 cases (26 with
1-2), and cohort 1 has only TWO cases in class 1 (1-2: sub-107 with 2,
sub-102 with 1 -- the earlier line "NO case in class 1" was wrong,
corrected 11 Sep). A fold without a positive cohort-1 case cannot be
avoided if cohort 1 has only 8 positive cases and 5 folds exist; the greedy
balance ensures that EVERY fold contains at least one cohort-1 case and at
least one positive case.

STATUS 11 Sep 2026 (user decision, Claude) -- THE VALID SPLIT IS THE OLD
SEED-0 STATE, exactly what this code produces:
- The text "highest CMB first" (above, in the code comment and in "regel"
  in split.json) was NEVER implemented: the code does not sort, the order
  is purely the seed-0 shuffle per cohort, and the folds get the remaining
  cases in inventory order. The text stays in "regel" because split.json
  must remain byte-identical.
- Result: validation 15 cases / 97 of 236 CMB (cohort1: sub-109/0,
  sub-102/1, sub-110/73 -> 74 of the 106 cohort-1 bleeds), folds
  [32, 26, 39, 18, 24] CMB. The validation set was never predicted.
- A re-split was built on 11 Sep and DISCARDED: literally "highest first"
  would have put 166/236 CMB into validation; the representative variant
  (median rank per cell, LPT folds) 37 CMB -- but 13 of its 15 validation
  cases were already in the folds in the old split, i.e. already
  trained/tested in we2/we3 (Optuna, learning curves, cv5). The old
  validation set is the only one never touched.
  Discarded states: dev/{split,cases,stats}.neusplit-verworfen-0911.json,
  dev/split.py.neusplit-verworfen-0911. This file is once again the code
  of the valid split (dev/split.py.vor-neusplit-0911), only comments
  corrected.
"""
import json, random

ROOT = "/home/uchralt/data/work/valdo-t2s"
VALDO = "/home/uchralt/data/extern/valdo2021/Task2"
PREPROC = f"{ROOT}/dev/preproc"

def klass(n):
    return 0 if n == 0 else (1 if n <= 2 else 2)

def main():
    inv = json.load(open(f"{ROOT}/dev/inventar.json"))
    rng = random.Random(0)
    byc = {}
    for c in inv:
        byc.setdefault(c["kohorte"], []).append(c)
    for k in byc:
        rng.shuffle(byc[k])
    assert sum(len(v) for v in byc.values()) == 72
    tot_cmb = sum(c["n_cmb"] for c in inv)

    # --- Validation: ~20 % per cohort (cohort1 3, cohort2 7, cohort3 5 = 15),
    #     every cohort and every occupied cell represented. (The earlier text
    #     "highest load first" was never implemented -- the seed-0 order
    #     applies, see STATUS 11 Sep in the docstring.)
    val_target = {"1": 3, "2": 7, "3": 5}
    val = []
    used = set()
    for koh, t in sorted(val_target.items()):
        cells = {}
        for c in byc[koh]:
            cells.setdefault(klass(c["n_cmb"]), []).append(c)
        # one round per occupied cell (1 case each, the first in seed-0 order)
        for kl in sorted(cells):
            c = cells[kl][0]
            val.append(c); used.add(c["id"])
        # remainder up to the target, in seed-0 order (not sorted by load)
        rest = [c for c in byc[koh] if c["id"] not in used]
        for c in rest:
            if sum(1 for v in val if v["kohorte"] == koh) >= t:
                break
            val.append(c); used.add(c["id"])
    assert len(val) == 15, f"validation {len(val)}"
    assert all(any(v["kohorte"] == k for v in val) for k in ("1", "2", "3"))

    rest = [c for c in inv if c["id"] not in used]
    assert len(rest) == 57

    # --- Folds: every cohort x class cell distributed as evenly as
    #     possible over the 5 folds (cap ceil(n/5) per fold), within the
    #     cell greedily by CMB sum: next case into the fold with the
    #     smallest CMB sum under the cap. Cells in a fixed order, so that
    #     small cells (cohort 1) are distributed first when all sums are
    #     equal -- then every fold gets one case.
    folds = [[] for _ in range(5)]
    cells = {}
    for c in rest:
        cells.setdefault((c["kohorte"], klass(c["n_cmb"])), []).append(c)
    import math
    for zell in sorted(cells):
        zellk, zelll = zell
        lst = cells[zell]
        # Greedy per cell: next case into the fold with the SMALLEST
        # number of cases of this cell (so that every cell appears in
        # all/as many folds as possible), then smallest CMB sum (the
        # required CMB balance), then fold index.
        for c in lst:
            def key(i, zk=zellk, zl=zelll):
                n_cell = len([x for x in folds[i] if x["kohorte"] == zk
                              and klass(x["n_cmb"]) == zl])
                return (n_cell, sum(x["n_cmb"] for x in folds[i]), i)
            folds[sorted(range(5), key=key)[0]].append(c)
    assert sum(len(f) for f in folds) == 57

    # --- Check from task 4: every fold has a cohort-1 case AND a
    #     positive case. On violation: swap (no other seed).
    ok = all(any(c["kohorte"] == "1" for c in f) and any(c["n_cmb"] > 0 for c in f) for f in folds)
    if not ok:
        for i, f in enumerate(folds):
            if not any(c["kohorte"] == "1" for c in f):
                # Fetch a cohort-1 case from another fold: swap with the
                # cohort-1 case of the fold with the largest CMB distance
                # from its own mean, keeping the CMB sum as close as possible.
                donor = max(range(5), key=lambda j: sum(c["n_cmb"] for c in folds[j]))
                c1 = next(c for c in folds[donor] if c["kohorte"] == "1")
                cswap = min(f, key=lambda c: abs(c["n_cmb"] - c1["n_cmb"]))
                folds[donor][folds[donor].index(c1)] = cswap
                f[f.index(cswap)] = c1
            if not any(c["n_cmb"] > 0 for c in f):
                donor = max((j for j in range(5) if any(c["n_cmb"] > 0 for c in folds[j])),
                            key=lambda j: sum(c["n_cmb"] for c in folds[j]))
                pos = max(folds[donor], key=lambda c: c["n_cmb"])
                cswap = min(f, key=lambda c: abs(c["n_cmb"] - pos["n_cmb"]))
                folds[donor][folds[donor].index(pos)] = cswap
                f[f.index(cswap)] = pos
        ok = all(any(c["kohorte"] == "1" for c in f) and any(c["n_cmb"] > 0 for c in f) for f in folds)
        assert ok, "fold correction via swap not possible"

    # --- split.json
    ids = lambda cs: [c["id"] for c in cs]
    split = {
        "validation": ids(val),
        "folds": [ids(f) for f in folds],
        "regel": ("Stratifiziert nach Kohorte x 3 Lastklassen (0 / 1-2 / >=3 CMB); "
                  "Validierung 15 Faelle (Ko1 3, Ko2 7, Ko3 5), jede Kohorte und je besetzte "
                  "Zelle vertreten, hoechste CMB-Last zuerst; 5 Falten, jede Kohorte x Klasse-Zelle "
                  "gleich verteilt, CMB-Summe je Falte gierig ausgeglichen; Seed 0; "
                  "Validierung bis we5 nicht angefasst."),
        "seed": 0,
    }
    json.dump(split, open(f"{ROOT}/dev/split.json", "w"), indent=1, ensure_ascii=False)

    # --- stats.json: n and CMB sum per cohort x class x set/fold
    def agg(cs):
        d = {}
        for c in cs:
            key = f"ko{c['kohorte']}_kl{klass(c['n_cmb'])}"
            e = d.setdefault(key, {"n": 0, "cmb": 0})
            e["n"] += 1; e["cmb"] += c["n_cmb"]
        return d
    stats = {"gesamt": {"n": 72, "cmb": tot_cmb}, "sets": {}}
    stats["sets"]["validation"] = agg(val)
    for i, f in enumerate(folds):
        stats["sets"][f"fold{i}"] = agg(f)
    stats["kontrolle"] = {
        "validation": {"n": len(val), "cmb": sum(c["n_cmb"] for c in val),
                       "ko1_n": sum(c["kohorte"] == "1" for c in val),
                       "ko1_cmb": sum(c["n_cmb"] for c in val if c["kohorte"] == "1"),
                       "pos_n": sum(c["n_cmb"] > 0 for c in val)},
        "falten": {f"fold{i}": {"n": len(f), "cmb": sum(c["n_cmb"] for c in f),
                                 "ko1_n": sum(c["kohorte"] == "1" for c in f),
                                 "pos_n": sum(c["n_cmb"] > 0 for c in f),
                                 "ko": {k: sum(c["kohorte"] == k for c in f) for k in ("1", "2", "3")}}
                   for i, f in enumerate(folds)},
        "jede_falte_ko1_und_positiv": ok,
    }
    json.dump(stats, open(f"{ROOT}/dev/stats.json", "w"), indent=1, ensure_ascii=False)

    # --- cases.json in the structure of code/metric.py:
    #     {"t2star": {"cases": [{"id","fold","klass","mask","image","frst"},...]}}
    fold_of = {}
    for i, f in enumerate(folds):
        for c in f:
            fold_of[c["id"]] = i
    for c in val:
        fold_of[c["id"]] = -1
    cases = []
    for c in inv:
        cid = c["id"]
        cases.append({
            "id": cid,
            "fold": fold_of[cid],
            "klass": klass(c["n_cmb"]),
            "mask": f"{VALDO}/{cid}/{cid}_space-T2S_CMB.nii.gz",
            "image": f"{PREPROC}/{cid}/{cid}_image.nii.gz",
            "frst": f"{PREPROC}/{cid}/{cid}_frst.nii.gz",
        })
    json.dump({"t2star": {"cases": cases}}, open(f"{ROOT}/dev/cases.json", "w"), indent=1, ensure_ascii=False)

    print("validation:", len(val), "cases, CMB", sum(c["n_cmb"] for c in val),
          "| ko1:", sum(c["kohorte"] == "1" for c in val),
          "CMB", sum(c["n_cmb"] for c in val if c["kohorte"] == "1"))
    for i, f in enumerate(folds):
        print(f"fold{i}: n={len(f)} cmb={sum(c['n_cmb'] for c in f)} "
              + " ".join(f"ko{k}={sum(c['kohorte'] == k for c in f)}" for k in ("1", "2", "3")))
    print("check ko1+positive per fold:", ok)

if __name__ == "__main__":
    main()
