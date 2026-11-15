#!/usr/bin/env python
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
# wk11_05_split.py -- step 5 of the wk11 plan (10.11.2026), check 1: the stage-1 / stage-2 split mismatch.
# WHAT: aggregates only, no case ids. For both sequences:
#       (a) how many scans sit in a different fold in the stage-1 split than in the stage-2 (patient-grouped) split;
#       (b) for every test fold k, how many candidates of the stage-2 TRAINING pool come from scans whose fold
#           assignment in the stage-1 split is != k while the patient-grouped split puts them in fold k
#           (= a patient trained on one sequence and tested on the other); and the mirror count (training scans
#           that the stage-1 split took away from the training pool);
#       (c) what fold assignment the candidate files that stage 2 was trained on actually carry, and whether the
#           stage-2 classifier of Phases 2/3 was ever trained on institutional candidates at all.
#       No new training; reads files already on disk; writes no result file.
# WHERE: input  MB/folds_frozen/{swi_stage1,swi_stage2,t2star}.json
#                 MB/gitter/manifest_{swi,swi_v2,swi_mix,t2star}.json
#                 MB/stage2/a03-aniso-e60-cat-{t2s,swi}-weich-z40/candidates.json  (Phase-1 stage-2 pool)
#                 MB/paper/phase4/candidates/<mod>/*_fold*/candidates.json          (target sets of Phase 2/3)
#                 MB/stage2/a1v2/selection.json                                     (frozen classifier provenance)
#        output wk11_splitcheck.json
import collections
import glob
import json
import os

MB = _P
OUT = _R + "/paper/wk11_splitcheck.json"
FOLDS = [0, 1, 2, 3, 4]
GRID = {"t2star": f"{MB}/gitter/manifest_t2star.json",
        "swi_stage1": f"{MB}/gitter/manifest_swi_v2.json",
        "swi_stage2": f"{MB}/gitter/manifest_swi_mix.json"}
POOL = {"t2star": f"{MB}/stage2/a03-aniso-e60-cat-t2s-weich-z40/candidates.json",
        "swi": f"{MB}/stage2/a03-aniso-e60-cat-swi-weich-z40/candidates.json"}


def foldmap_json(path):
    d = json.load(open(path))
    return {k: int(v) for k, v in d["folds"].items()}


def foldmap_manifest(path):
    return {e["id"]: int(e["fold"]) for e in json.load(open(path))}


def agree(a, b):
    k = set(a) & set(b)
    return sum(1 for i in k if a[i] == b[i]), len(k)


def main():
    out = {}
    s1 = foldmap_json(f"{MB}/folds_frozen/swi_stage1.json")
    s2 = foldmap_json(f"{MB}/folds_frozen/swi_stage2.json")
    t = foldmap_json(f"{MB}/folds_frozen/t2star.json")
    g = {"swi_stage1": foldmap_manifest(GRID["swi_stage1"]),
         "swi_stage2": foldmap_manifest(GRID["swi_stage2"]),
         "t2star": foldmap_manifest(GRID["t2star"])}

    # (a) the two SWI splits, and what each phase actually reads
    same, n = agree(s1, s2)
    out["splits"] = dict(
        swi_scans=n,
        swi_same_fold=same,
        swi_different_fold=n - same,
        t2star_one_split_only=True,
        grid_swi_v2_equals_stage1_split=agree(g["swi_stage1"], s1) == (61, 61),
        grid_swi_mix_equals_stage2_split=agree(g["swi_stage2"], s2) == (61, 61),
        t2star_frozen_equals_grid=agree(t, g["t2star"]) == (75, 75),
        which_split_phase234_reads="stage-1 split (gitter/manifest_*_v2 == folds_frozen/swi_stage1.json)",
    )

    # (b) per test fold k of the split that Phases 2/3/4 read (stage-1 split for SWI):
    #     how many stage-2 training candidates come from a scan that the patient-grouped split puts in fold k.
    #     The patient-grouped split for SWI is swi_stage2 (= manifest_swi_mix, twins moved to the T2* fold).
    pool = {m: json.load(open(p)) for m, p in POOL.items()}
    ncand = {m: collections.Counter(x["case"] for x in pool[m]["candidates"]) for m in pool}
    per_fold = {}
    tot_leak = tot_mirror = 0
    for k in FOLDS:
        train = [i for i, f in s1.items() if f != k]          # stage-2 training scans of fold k, as run
        leak = [i for i in train if s2.get(i) == k]           # same patient belongs to test fold k
        mirr = [i for i, f in s1.items() if f == k and s2.get(i) != k]   # own test scans the run held out too
        e_leak = sum(ncand["swi"][i] for i in leak)
        e_mirr = sum(ncand["swi"][i] for i in mirr)
        tot_leak += e_leak
        tot_mirror += e_mirr
        per_fold[str(k)] = dict(
            test_scans_as_run=sum(1 for f in s1.values() if f == k),
            train_scans_as_run=len(train),
            train_scans_of_a_test_fold_patient=len(leak),
            train_candidates_of_a_test_fold_patient=e_leak,
            own_test_scans_held_out_extra=len(mirr),
            own_test_candidates_held_out_extra=e_mirr,
        )
    swi_pool_total = sum(ncand["swi"].values())
    out["swi_stage2_pool_vs_patient_split"] = dict(
        pool_candidates=swi_pool_total,
        per_test_fold=per_fold,
        leaked_candidates_all_folds=tot_leak,
        leaked_share=round(tot_leak / swi_pool_total, 4) if swi_pool_total else None,
        mirror_candidates_all_folds=tot_mirror,
        scans_in_a_different_fold=n - same,
    )
    # T2*: the frozen split and the split used by every phase are the same -> no such count possible
    out["t2star_pool_vs_split"] = dict(
        pool_candidates=sum(ncand["t2star"].values()),
        pool_foldmap_equals_frozen_split=agree({k2: int(v["fold"]) for k2, v in pool["t2star"]["cases"].items()},
                                               t) == (75, 75),
        leaked_candidates_all_folds=0,
    )
    # fold recorded inside the candidate files (what stage 2 sees) vs the two splits
    for m in ("swi", "t2star"):
        pm = {k2: int(v["fold"]) for k2, v in pool[m]["cases"].items()}
        out.setdefault("pool_fold_provenance", {})[m] = dict(
            equals_stage1_split=agree(pm, s1 if m == "swi" else t) == (len(pm), len(pm)),
            equals_patient_split=agree(pm, s2 if m == "swi" else t) == (len(pm), len(pm)),
        )

    # (c) was the stage-2 classifier of Phases 2/3 ever trained on institutional candidates?
    sel = json.load(open(f"{MB}/stage2/a1v2/selection.json"))
    out["phase23_stage2_provenance"] = dict(
        winner_arm=sel["winner"],
        trained_on=sel["frozen_classifier"]["training"],
        hospital_data_in_training=bool(sel["training_data"].get("no_hospital_data")) is False,
        valdo_cases=sel["training_data"]["count"],
    )
    # the phase-4 target candidate sets: do any of them contain candidates of a fold other than the block's own?
    tsets = {}
    for mod in ("t2star", "swi"):
        sets = sorted(glob.glob(f"{MB}/paper/phase4/candidates/{mod}/*/candidates.json"))
        mixed = 0
        for p in sets:
            c = json.load(open(p))
            if len({int(v["fold"]) for v in c["cases"].values()}) > 1:
                mixed += 1
        tsets[mod] = dict(n_sets=len(sets), sets_with_more_than_one_fold=mixed)
    out["phase4_target_sets"] = tsets

    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
