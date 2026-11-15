"""Patient-level operating point for the anti-amyloid threshold (> 4 CMB excludes), user yes 01.11.2026.

The chain's classifier threshold was chosen for lesion F1. For the patient decision "> 4 CMB?" it is chosen instead, per
outer fold k, on the candidates of the other four folds (their scores are out-of-fold already): the threshold with the
highest specificity among those keeping sensitivity >= SENS_MIN on those folds (ties -> the lower threshold). Applied
unchanged to fold k. Variant "cut": keep the lesion-F1 decisions of the chain, shift the COUNT cut-off c (predicted count
> c) on the other folds the same way. Counts are per seed (accepted candidates of that seed), then averaged over seeds.
Reference count per patient = n_cmb of the manifest (26-connected components). Bootstrap over patients (2000 draws).

    python patient_threshold.py              # -> patient_threshold_summary.json (aggregates only)
    python patient_threshold.py --selftest   # synthetic check, no data
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
import argparse, json, sys
import numpy as np

MB = _P
DUMPS = [f"{MB}/paper/decisions/chainA_s{s}.jsonl" for s in (0, 1)]
MODE = "pooled tset-plain"
BAND = (3, 8)            # step 2 (pre-specified): predicted counts 3..8 go to a radiologist, who is assumed to decide correctly
GROUP = {**{l: "cortex" for l in (3, 42)}, **{l: "cerebral white matter" for l in (2, 41)},
         **{l: "deep grey matter" for l in (10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60)},
         **{l: "CSF / ventricles" for l in (4, 5, 14, 15, 24, 43, 44)}, **{l: "cerebellum" for l in (7, 8, 46, 47)},
         16: "brainstem", 0: "outside labels"}
MANIFEST = {"valdo": _R + "/dev/manifest_valdo_gitter_d.json",
            "t2star": f"{MB}/gitter/manifest_t2star.json", "swi": f"{MB}/gitter/manifest_swi_mix_v2.json"}   # stage-2 folds as in the dump
CUT = 4
SENS_MIN = 0.95
GRID = np.round(np.arange(0.05, 0.991, 0.01), 2)


def sens_spec(ref, pred):
    ref, pred = np.asarray(ref) > CUT, np.asarray(pred)
    tp = int((ref & pred).sum()); fn = int((ref & ~pred).sum()); tn = int((~ref & ~pred).sum()); fp = int((~ref & pred).sum())
    return dict(tp=tp, fn=fn, tn=tn, fp=fp, sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                wrongly_eligible=fn, wrongly_excluded=fp)


def choose(values, ref_counts, count_of):
    """values: candidate operating points; count_of(v) -> predicted counts of the training patients."""
    best = None
    for v in values:
        r = sens_spec(ref_counts, np.asarray(count_of(v)) > CUT)
        if r["sensitivity"] >= SENS_MIN - 1e-9 and (best is None or r["specificity"] > best[1] + 1e-12):
            best = (v, r["specificity"])
    return best[0] if best else values[0]


def analyse(cases, n_boot=2000, seed=20261101):
    """cases: list of dict(cohort, fold, ref, scores=[array per seed], accepted=[int per seed])."""
    out = {}
    pred_thr, pred_cut, pred_base, pred_ens = [], [], [], []
    for k in sorted({c["fold"] for c in cases}):
        tr = [c for c in cases if c["fold"] != k]; te = [c for c in cases if c["fold"] == k]
        cnt = lambda cs, t: [np.mean([(s >= t).sum() for s in c["scores"]]) for c in cs]
        t = choose(GRID, [c["ref"] for c in tr], lambda v: cnt(tr, v))
        base = lambda cs: [np.mean(c["accepted"]) for c in cs]
        cut = choose(list(range(CUT, 31)), [c["ref"] for c in tr], lambda v: [b - (v - CUT) for b in base(tr)])   # pred: count > v
        ens = lambda cs, t: [(np.mean(np.vstack(c["scores"]), axis=0) >= t).sum() if len(c["scores"][0]) else 0 for c in cs]
        te_ = choose(GRID, [c["ref"] for c in tr], lambda v: ens(tr, v))       # step 4 cheap: mean score of the two seeds
        for c, pt, pb, pe in zip(te, cnt(te, t), base(te), ens(te, te_)):
            pred_thr.append((c, pt > CUT, t, pt)); pred_base.append((c, pb > CUT, None, pb)); pred_cut.append((c, pb > cut, cut, pb))
            pred_ens.append((c, pe > CUT, te_, pe))
    rng = np.random.RandomState(seed)
    for scope in ["all"] + sorted({c["cohort"] for c in cases}):
        sel = lambda L: [x for x in L if scope == "all" or x[0]["cohort"] == scope]
        block = {}
        for name, L in (("lesion_f1_operating_point", pred_base), ("patient_threshold", pred_thr), ("count_cut", pred_cut),
                        ("seed_ensemble_patient_threshold", pred_ens)):
            S = sel(L); ref = [x[0]["ref"] for x in S]; pr = [x[1] for x in S]
            r = sens_spec(ref, pr); boots = []
            for _ in range(n_boot):
                ix = rng.randint(len(S), size=len(S)); rr = sens_spec([ref[i] for i in ix], [pr[i] for i in ix])
                boots.append((rr["sensitivity"], rr["specificity"]))
            b = np.array(boots)
            r.update(n=len(S), sensitivity_ci=[round(float(np.percentile(b[:, 0], 2.5)), 3), round(float(np.percentile(b[:, 0], 97.5)), 3)],
                     specificity_ci=[round(float(np.percentile(b[:, 1], 2.5)), 3), round(float(np.percentile(b[:, 1], 97.5)), 3)])
            if name != "lesion_f1_operating_point":
                r["chosen_per_fold"] = sorted({x[2] for x in S})
            inband = [BAND[0] <= x[3] <= BAND[1] for x in S]                      # step 2: radiologist confirms the band
            eff = [(x[0]["ref"] > CUT) if b else x[1] for x, b in zip(S, inband)]
            rb = sens_spec(ref, eff)
            r["with_radiologist_band"] = dict(band=list(BAND), share_to_radiologist=round(float(np.mean(inband)), 3),
                                              sensitivity=round(rb["sensitivity"], 3), specificity=round(rb["specificity"], 3),
                                              wrongly_eligible=rb["wrongly_eligible"], wrongly_excluded=rb["wrongly_excluded"])
            r["sensitivity"] = round(r["sensitivity"], 3); r["specificity"] = round(r["specificity"], 3)
            block[name] = r
        out[scope] = block
    return out


BAND_ALTERNATIVES = [(2, 6), (3, 8), (4, 10), (2, 10)]   # reviewer check 01.11.: is the pre-specified 3..8 a lucky choice?
BAND_GRID = [(lo, hi) for lo in range(1, 5) for hi in range(5, 13)]
BAND_TARGET = 0.95       # nested band: per outer fold, the band with the SMALLEST radiologist share whose automatic decisions
                         # reach sensitivity AND specificity >= BAND_TARGET on the other four folds (fallback: widest band)


def band_effect(S, band):
    ref = [c["ref"] for c, _ in S]; inb = [band[0] <= n <= band[1] for _, n in S]
    eff = [(c["ref"] > CUT) if b else (n > CUT) for (c, n), b in zip(S, inb)]
    r = sens_spec(ref, eff)
    return dict(band=list(band), share_to_radiologist=round(float(np.mean(inb)), 3), sensitivity=round(r["sensitivity"], 3),
                specificity=round(r["specificity"], 3), wrongly_eligible=r["wrongly_eligible"], wrongly_excluded=r["wrongly_excluded"])


def band_analysis(cases):
    """Lesion-F1 operating point (mean accepted count over seeds); fixed alternative bands and a band chosen per outer fold."""
    S_all = [(c, float(np.mean(c["accepted"]))) for c in cases]
    nested, chosen = [], {}
    for k in sorted({c["fold"] for c in cases}):
        tr = [x for x in S_all if x[0]["fold"] != k]; te = [x for x in S_all if x[0]["fold"] == k]
        ok = [b for b in BAND_GRID if (lambda r: r["sensitivity"] >= BAND_TARGET and r["specificity"] >= BAND_TARGET)(band_effect(tr, b))]
        best = min(ok, key=lambda b: (band_effect(tr, b)["share_to_radiologist"], b[1] - b[0])) if ok else (1, 12)
        chosen[k] = list(best); nested += [(x, best) for x in te]
    out = {}
    for scope in ["all"] + sorted({c["cohort"] for c in cases}):
        sel = [x for x in S_all if scope == "all" or x[0]["cohort"] == scope]
        out[scope] = dict(fixed={f"{lo}-{hi}": band_effect(sel, (lo, hi)) for lo, hi in BAND_ALTERNATIVES})
        nz = [(x, b) for x, b in nested if scope == "all" or x[0]["cohort"] == scope]
        inb = [b[0] <= n <= b[1] for (c, n), b in nz]
        eff = [(c["ref"] > CUT) if i else (n > CUT) for ((c, n), _), i in zip(nz, inb)]
        r = sens_spec([c["ref"] for (c, _), _ in nz], eff)
        out[scope]["nested"] = dict(share_to_radiologist=round(float(np.mean(inb)), 3), sensitivity=round(r["sensitivity"], 3),
                                    specificity=round(r["specificity"], 3), wrongly_eligible=r["wrongly_eligible"],
                                    wrongly_excluded=r["wrongly_excluded"])
    out["nested_band_per_fold"] = chosen
    out["rule"] = (f"nested: per outer fold the band (lo 1-4, hi 5-12) with the smallest radiologist share reaching sensitivity and "
                   f"specificity >= {BAND_TARGET} on the other four folds; folds pooled over cohorts as in the main analysis")
    return out


def load():
    man, mfold = {}, {}
    for coh, f in MANIFEST.items():
        for e in json.load(open(f)):
            man[(coh, e["id"])] = e["n_cmb"]; mfold[(coh, e["id"])] = e["fold"]
    per = {}; fpa = {}
    for si, path in enumerate(DUMPS):
        for l in open(path):
            r = json.loads(l)
            if r["mode"] != MODE:
                continue
            c = per.setdefault((r["cohort"], r["case"]), dict(cohort=r["cohort"], fold=r["fold"], scores=[[], []], accepted=[0, 0]))
            c["scores"][si].append(r["score"]); c["accepted"][si] += int(r["accepted"])
            if r["accepted"]:                                                     # step 3: error analysis, aggregates only
                kind = "true" if r["touched"] else "false"
                g = GROUP.get(r["tissue"], "other")
                size = "1-3 vox" if r["voxels"] <= 3 else "4-10 vox" if r["voxels"] <= 10 else "11-40 vox" if r["voxels"] <= 40 else ">40 vox"
                for key2 in (("tissue", g), ("size", size)):
                    d = fpa.setdefault(r["cohort"], {}).setdefault(key2[0], {}).setdefault(key2[1], {"true": 0, "false": 0})
                    d[kind] += 1
    cases = []
    for key, ref in man.items():
        c = per.get(key, dict(cohort=key[0], fold=mfold[key], scores=[[], []], accepted=[0, 0]))   # no candidate: count 0
        c["ref"] = ref; c["scores"] = [np.array(s, float) for s in c["scores"]]; cases.append(c)
    return cases, sum(1 for k in man if k not in per), fpa


def selftest():
    rng = np.random.RandomState(0); cases = []
    for i in range(200):
        ref = int(rng.poisson(3)); fps = int(rng.poisson(3))
        sc = [np.concatenate([rng.uniform(0.5, 1.0, ref), rng.uniform(0.3, 0.6, fps)]) for _ in (0, 1)]
        cases.append(dict(cohort="x", fold=i % 5, ref=ref, scores=sc, accepted=[int((s >= 0.35).sum()) for s in sc]))
    r = analyse(cases, n_boot=50)["all"]
    assert r["patient_threshold"]["specificity"] > r["lesion_f1_operating_point"]["specificity"], r
    assert r["patient_threshold"]["sensitivity"] >= 0.8, r
    print("selftest ok:", {k: (v["sensitivity"], v["specificity"]) for k, v in r.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--chain", default="chainA", help="dump prefix in paper/decisions (chainA, arm_tset-tta8, ...)")
    ap.add_argument("--mode", default=MODE); a = ap.parse_args()
    DUMPS[:] = [f"{MB}/paper/decisions/{a.chain}_s{s}.jsonl" for s in (0, 1)]; MODE = a.mode
    if a.selftest:
        selftest(); sys.exit(0)
    cases, missing, fpa = load()
    res = analyse(cases)
    res["radiologist_band_robustness"] = band_analysis(cases)
    res["false_positive_analysis"] = dict(note="accepted candidates at the lesion-F1 operating point, summed over the seeds; "
                                               "tissue = SynthSeg label at the candidate centre", by_cohort=fpa)
    res["meta"] = dict(rule=f"exclusion if > {CUT} CMB", sens_min=SENS_MIN, mode=MODE, dumps=DUMPS,
                       patients_without_any_candidate=missing,
                       note="aggregates only; operating point chosen per outer fold on the other four folds")
    out = f"{MB}/paper/patient_threshold_summary.json" if a.chain == "chainA" else f"{MB}/paper/patient_threshold_{a.chain}.json"
    res["meta"]["chain"] = a.chain
    json.dump(res, open(out, "w"), indent=1)
    for scope in ("all", "t2star", "swi", "valdo"):
        print(scope, {k: (v["sensitivity"], v["specificity"], v["wrongly_excluded"]) for k, v in res[scope].items()})
