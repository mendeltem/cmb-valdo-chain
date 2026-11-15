"""wk3: Leave-One-Dataset-Out (LODO) for the CMB paper Phase III.

Sites (five):
  SABRE, RSS, ALFA  -- from VALDO sub-ids (2nd digit: 1=SABRE, 2=RSS, 3=ALFA)
  t2star, swi       -- in-house T2*GRE and SWI

For each held-out site D:
  (a) threshold transferred: nested CV on training sites -> threshold applied to D
  (b) threshold calibrated:  nested CV on training sites, then threshold re-tuned on D's own training folds only
  Baseline: in-dataset S/P (stage-2 trained on ALL sites including D, standard nested CV)

III-C: site-count curve k=1..4
III-C fixed: same but pool subsampled to N_fix

All CPU (feature models: logistic, forest, boosting).
"""
from __future__ import annotations
import json, os, sys, time, itertools, numpy as np
from typing import Dict, List

from cmb.paths import ROOT
from cmb.paths import MB
sys.path.insert(0, ROOT)

from cmb.stage2.classifier import features_of, fit_predict_features, THRESHOLDS, lesion_scores

SITES = {
    "valdo": os.path.join(ROOT, "ergebnisse/stage2/a03-aniso-e60-gd-weich-z40"),
    "t2star": os.path.join(MB, "stage2/a03-aniso-e60-cat-t2s-weich-z40"),
    "swi":    os.path.join(MB, "stage2/a03-aniso-e60-cat-swi-weich-z40"),
}

# F1: patients scanned in both own cohorts ("twins"), swi id -> t2star id. The paper had seven. The ids are private:
# they are read from the JSON file named by CMB_TWINS ({"<swi id>": "<t2star id>", ...}); without it no case is excluded.
from cmb.paths import TWINS as _TWINS
TWIN_MAP = json.load(open(_TWINS)) if _TWINS else {}
# Reverse: t2star -> swi
TWIN_MAP_R = {v: k for k, v in TWIN_MAP.items()}

def twin_exclusion(held_out_site, case_id):
    """Return True if this case should be EXCLUDED from the training pool
    because of F1 (twins in both in-house datasets)."""
    if held_out_site == "swi" and case_id in TWIN_MAP:
        # held-out is SWI: remove the twin T2* case from T2* training pool
        return case_id in TWIN_MAP  # the twin IS the swi case; its t2s partner is TWIN_MAP[case_id]
    if held_out_site == "t2star" and case_id in TWIN_MAP_R:
        # held-out is T2*: remove the twin SWI case from SWI training pool
        return case_id in TWIN_MAP_R
    return False

def twin_excluded_ids(held_out_site):
    """Return the set of case IDs to EXCLUDE from training when held_out_site is held out."""
    if held_out_site == "swi":
        # Remove T2* twins from the T2* training pool
        return set(TWIN_MAP.values())
    if held_out_site == "t2star":
        # Remove SWI twins from the SWI training pool
        return set(TWIN_MAP.keys())
    return set()

def valdo_site(sub_id: str) -> str:
    """sub-104 -> 'SABRE', sub-2xx -> 'RSS', sub-3xx -> 'ALFA'"""
    digits = sub_id.split("-")[1]
    return {"1": "SABRE", "2": "RSS", "3": "ALFA"}.get(digits[0], "UNKNOWN")

def load_site(name: str) -> tuple:
    """Load candidates + features for one site. Returns (rows, x, cases)."""
    info = json.load(open(f"{SITES[name]}/candidates.json"))
    patches = np.load(f"{SITES[name]}/patches.npy", mmap_mode="r")
    threshold = info["threshold"]
    rows, feats, cases = [], [], {}
    for case_id, case in info["cases"].items():
        if name == "valdo":
            site = valdo_site(case_id)
        else:
            site = name
        cases[case_id] = dict(fold=case["fold"], n_reference=case["n_reference"], site=site)
    for i, row in enumerate(info["candidates"]):
        site = cases[row["case"]]["site"]
        rows.append(dict(row, fold=cases[row["case"]]["fold"], site=site, case=row["case"]))
        feats.append(features_of(np.asarray(patches[i]), row, threshold))
    print(f"  {name}: {len(rows)} candidates, {len(cases)} cases, "
          f"positives={sum(1 for r in rows if r['touched'])}", flush=True)
    return rows, np.asarray(feats, np.float32), cases

def load_all():
    """Load all sites. Returns (rows, x, cases) for the FULL pool plus a site list."""
    all_rows, all_feats, all_cases = [], [], {}
    for name in SITES:
        rows, x, cases = load_site(name)
        all_rows.extend(rows)
        all_feats.extend(x)
        all_cases.update(cases)
    x = np.asarray(all_feats, np.float32)
    sites = sorted({r["site"] for r in all_rows})
    return all_rows, x, all_cases, sites

def nested_cv_score(all_rows, all_x, all_cases, model_kind, seeds, train_sites, eval_site):
    """
    Nested CV: train classifier on candidates of train_sites (all folds),
    threshold chosen on out-of-fold scores within the training folds,
    evaluate unchanged on eval_site. Returns list of per-seed score dicts.
    F1: twin cases of the OTHER in-house dataset are removed from the training pool.
    """
    results = []
    excluded = twin_excluded_ids(eval_site)
    site_of = np.array([r["site"] for r in all_rows])
    train_mask = np.isin(site_of, list(train_sites)) & ~np.array([r["case"] in excluded for r in all_rows])
    test_mask  = site_of == eval_site
    train_idx = np.nonzero(train_mask)[0]
    test_idx  = np.nonzero(test_mask)[0]
    tx, txx = all_x[train_idx], all_x[test_idx]
    train_rows_i = [all_rows[i] for i in train_idx]
    test_rows_i  = [all_rows[i] for i in test_idx]
    train_folds = np.array([r["fold"] for r in train_rows_i])
    yp = np.array([1 if r["touched"] else 0 for r in train_rows_i])
    train_cases = {c: v for c, v in all_cases.items() if v["site"] in train_sites}
    fit_predict = fit_predict_features(model_kind)
    for seed in seeds:
        # Inner loop: OOF scores within the training folds -> acceptance threshold
        n_folds = int(train_folds.max()) + 1
        oof = np.full(len(train_rows_i), np.nan)
        for j in range(n_folds):
            it, ih = train_folds != j, train_folds == j
            if ih.sum() == 0 or it.sum() < 5:
                continue
            oof[ih] = fit_predict(tx[it], yp[it], tx[ih], seed=seed + j)
        valid = ~np.isnan(oof)
        thr_rows = [train_rows_i[i] for i in np.nonzero(valid)[0]]
        thr_oof  = oof[valid]
        thr_cases = {c: v for c, v in train_cases.items() if any(r["case"] == c for r in thr_rows)}
        threshold = float(max(THRESHOLDS, key=lambda t: lesion_scores(thr_rows, thr_oof >= t, thr_cases)["f1"]))
        test_scores = fit_predict(tx, yp, txx, seed=seed)
        accepted = test_scores >= threshold
        eval_cases = {c: v for c, v in all_cases.items() if v["site"] == eval_site}
        sc = lesion_scores(test_rows_i, accepted, eval_cases)
        sc["threshold"] = threshold
        sc["seed"] = seed
        results.append(sc)
    return results

def in_dataset_cv(all_rows, all_x, all_cases, model_kind, seeds, site):
    """In-dataset S/P baseline: pooled stage 2 over ALL sites; outer CV over the folds.
    For fold k the classifier sees every candidate (all sites) with fold != k, so it
    HAS seen site D's training folds. Score on D's fold-k candidates only."""
    site_of = np.array([r["site"] for r in all_rows])
    folds   = np.array([r["fold"] for r in all_rows])
    d_idx = np.nonzero(site_of == site)[0]
    d_folds = np.unique(folds[d_idx])
    fit_predict = fit_predict_features(model_kind)
    results = []
    for seed in seeds:
        # pooled OOF over the global folds (threshold chosen where D's folds live)
        n_folds = int(folds.max()) + 1
        oof = np.full(len(all_rows), np.nan)
        yp = np.array([1 if r["touched"] else 0 for r in all_rows])
        for j in range(n_folds):
            it, ih = folds != j, folds == j
            if ih.sum() == 0 or it.sum() < 5:
                continue
            oof[ih] = fit_predict(all_x[it], yp[it], all_x[ih], seed=seed + j)
        # threshold on OOF within the folds that D occupies
        d_oof_idx = d_idx[~np.isnan(oof[d_idx])]
        d_oof_rows = [all_rows[i] for i in d_oof_idx]
        d_oof = oof[d_oof_idx]
        d_cases = {c: v for c, v in all_cases.items() if v["site"] == site}
        threshold = float(max(THRESHOLDS, key=lambda t: lesion_scores(d_oof_rows, d_oof >= t, d_cases)["f1"]))
        # final model on all folds != k, applied to D's fold k; pool per fold
        per_seed = {}
        for k in d_folds:
            train = folds != k
            test = np.array([i for i in d_idx if folds[i] == k])
            if len(test) == 0:
                continue
            scores = fit_predict(all_x[train], yp[train], all_x[test], seed=seed)
            accepted = scores >= threshold
            trows = [all_rows[i] for i in test]
            fold_cases = {c: v for c, v in d_cases.items() if any(r["case"] == c for r in trows)}
            per_seed[k] = lesion_scores(trows, accepted, fold_cases)
        # pooled over D's folds
        tp = sum(v["tp"] for v in per_seed.values()); fp = sum(v["fp"] for v in per_seed.values())
        fn = sum(v["fn"] for v in per_seed.values()); nc = sum(v["fp_per_case"] for v in per_seed.values())
        sc = dict(f1=2*tp/max(2*tp+fp+fn,1), precision=tp/max(tp+fp,1),
                  sensitivity=tp/max(tp+fn,1), fp_per_case=float(nc)/len(d_cases), tp=tp, fp=fp, fn=fn)
        sc["threshold"] = threshold
        sc["seed"] = seed
        results.append(sc)
    return results

def mean_ci(results, key="f1"):
    vals = [r[key] for r in results]
    m = float(np.mean(vals))
    if len(vals) > 1:
        ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    else:
        ci = [m, m]
    return m, ci, vals

def lodo_calibrated(all_rows, all_x, all_cases, model_kind, seeds, train_sites, D):
    """III-A variant (b): 'cheap calibration' -- same classifier trained ONLY on the
    other sites, but the acceptance threshold is re-tuned on D's own out-of-fold
    scores (threshold only, no D data in the model)."""
    site_of = np.array([r["site"] for r in all_rows])
    excluded = twin_excluded_ids(D)
    train_mask = np.isin(site_of, list(train_sites)) & ~np.array([r["case"] in excluded for r in all_rows])
    test_mask = site_of == D
    train_idx = np.nonzero(train_mask)[0]; test_idx = np.nonzero(test_mask)[0]
    tx, txx = all_x[train_idx], all_x[test_idx]
    train_rows_i = [all_rows[i] for i in train_idx]
    test_rows_i  = [all_rows[i] for i in test_idx]
    yp = np.array([1 if r["touched"] else 0 for r in train_rows_i])
    d_cases = {c: v for c, v in all_cases.items() if v["site"] == D}
    fit_predict = fit_predict_features(model_kind)
    results = []
    for seed in seeds:
        scores = fit_predict(tx, yp, txx, seed=seed)
        # 06.11.2026 fix: the threshold was tuned on ALL of D and scored on the same candidates (leak). Now, per fold k of D:
        # tune on D's candidates with fold != k (the 'training folds' of the spec), score fold k, pool TP/FP/FN over k.
        d_fold = np.array([r["fold"] for r in test_rows_i]); tp = fp = fn = 0; thrs = []
        for k in np.unique(d_fold):
            cal, ev = d_fold != k, d_fold == k
            cal_rows = [test_rows_i[i] for i in np.nonzero(cal)[0]]; ev_rows = [test_rows_i[i] for i in np.nonzero(ev)[0]]
            cal_cases = {c: v for c, v in d_cases.items() if any(r["case"] == c for r in cal_rows)}
            ev_cases = {c: v for c, v in d_cases.items() if any(r["case"] == c for r in ev_rows)}
            thr = float(max(THRESHOLDS, key=lambda t: lesion_scores(cal_rows, scores[cal] >= t, cal_cases)["f1"]))
            s_k = lesion_scores(ev_rows, scores[ev] >= thr, ev_cases); tp += s_k["tp"]; fp += s_k["fp"]; fn += s_k["fn"]; thrs.append(thr)
        sc = dict(f1=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1), sensitivity=tp / max(tp + fn, 1),
                  fp_per_case=fp / max(len(d_cases), 1), tp=tp, fp=fp, fn=fn)
        sc["threshold"] = float(np.mean(thrs)); sc["seed"] = seed
        results.append(sc)
    return results

def run_phase3a(all_rows, all_x, all_cases, all_sites, model_kind="forest", seeds=[0,1,2], only=None) -> dict:
    """III-A: LODO over 5 sites, 3 seeds."""
    out = {}
    t0 = time.time()
    for D in (only or all_sites):
        others = [s for s in all_sites if s != D]
        print(f"\n=== III-A held-out: {D} (train on {others}) ===", flush=True)
        lodo_a = nested_cv_score(all_rows, all_x, all_cases, model_kind, seeds, set(others), D)
        lodo_b = lodo_calibrated(all_rows, all_x, all_cases, model_kind, seeds, set(others), D)
        ind = in_dataset_cv(all_rows, all_x, all_cases, model_kind, seeds, D)
        f1_a, ci_a, _ = mean_ci(lodo_a)
        f1_b, ci_b, _ = mean_ci(lodo_b)
        f1_ind, ci_ind, _ = mean_ci(ind)
        out[D] = dict(
            lodo_transfer=dict(f1=f1_a, ci95=ci_a, precision=mean_ci(lodo_a,"precision")[0],
                                sensitivity=mean_ci(lodo_a,"sensitivity")[0],
                                fp_per_case=mean_ci(lodo_a,"fp_per_case")[0],
                                threshold=mean_ci(lodo_a,"threshold")[0]),
            lodo_calibrated=dict(f1=f1_b, ci95=ci_b, threshold=mean_ci(lodo_b,"threshold")[0]),
            in_dataset=dict(f1=f1_ind, ci95=ci_ind),
            gap_transfer=f1_ind - f1_a,
            gap_calibrated=f1_ind - f1_b,
            n_train_candidates=sum(1 for r in all_rows if r["site"] in others),
            n_test_candidates=sum(1 for r in all_rows if r["site"] == D),
            twins_removed=len(twin_excluded_ids(D)),
            twins_removed_note="F1: twin cases of the other in-house dataset removed from the training pool",
        )
        print(f"  LODO(a) F1={f1_a:.3f} [{ci_a[0]:.3f},{ci_a[1]:.3f}]  LODO(b) F1={f1_b:.3f}  "
              f"in-dataset F1={f1_ind:.3f}  gap(a)={f1_ind-f1_a:.3f} gap(b)={f1_ind-f1_b:.3f}", flush=True)
    out["meta"] = dict(model=model_kind, seeds=seeds, duration_s=round(time.time()-t0,1),
                       sites=all_sites,
                       note="LODO: stage-2 trained only on the other sites; in-dataset = pooled stage 2 that saw D's training folds.")
    return out

def combos_for(others, k, rng_seed=42):
    if k == 2:
        import random
        all_combos = list(itertools.combinations(others, 2))
        random.seed(rng_seed)
        return random.sample(all_combos, min(6, len(all_combos)))
    return list(itertools.combinations(others, k))

def run_phase3c(all_rows, all_x, all_cases, all_sites, model_kind="forest", only=None) -> dict:
    """III-C: site-count curve k=1..4; 1 seed (best k=4 with 3 seeds)."""
    out = {}
    t0 = time.time()
    times = []
    for D in (only or all_sites):
        others = [s for s in all_sites if s != D]
        out[D] = {}
        for k in [1, 2, 3, 4]:
            if k > len(others):
                continue
            combos = combos_for(others, k)
            seeds = [0, 1, 2] if k == 4 else [0]
            all_results = []
            for combo in combos:
                t = time.time()
                r = nested_cv_score(all_rows, all_x, all_cases, model_kind, seeds, set(combo), D)
                times.append(time.time() - t)
                all_results.extend(r)
                print(f"  D={D} k={k} combo={combo} F1={mean_ci(r)[0]:.3f} ({times[-1]:.0f}s)", flush=True)
            f1, ci, _ = mean_ci(all_results)
            out[D][str(k)] = dict(f1=f1, ci95=ci, n_combos=len(combos), n_seeds=len(seeds))
    out["meta"] = dict(model=model_kind, duration_s=round(time.time()-t0,1),
                       mean_run_s=round(float(np.mean(times)),1) if times else 0,
                       n_runs=len(times))
    return out

def run_phase3c_fixed(all_rows, all_x, all_cases, all_sites, model_kind="forest", only=None) -> dict:
    """III-C fixed-size: for every k, subsample the k-site pool to N_fix candidates
    (smallest single-site pool among the training sites of D; equal shares of true/false,
    equal shares from the k sites). Same combinations and seeds as III-C."""
    from collections import Counter
    site_of = np.array([r["site"] for r in all_rows])
    y = np.array([1 if r["touched"] else 0 for r in all_rows])
    pool_size = Counter(site_of)
    out = {}
    t0 = time.time()
    for D in (only or all_sites):
        others = [s for s in all_sites if s != D]
        N_fix = min(pool_size[s] for s in others)
        out[D] = {"N_fix": N_fix}
        for k in [1, 2, 3, 4]:
            if k > len(others):
                continue
            combos = combos_for(others, k)
            seeds = [0, 1, 2] if k == 4 else [0]
            all_results = []
            excluded = twin_excluded_ids(D)
            for combo in combos:
                # build the fixed-size training pool
                per_site = N_fix // k
                idx_pool = []
                for s in combo:
                    s_idx = np.nonzero(site_of == s)[0]
                    s_idx = np.array([i for i in s_idx if all_rows[i]["case"] not in excluded])
                    pos = s_idx[y[s_idx] == 1]; neg = s_idx[y[s_idx] == 0]
                    npos = int(round(per_site * len(pos) / max(len(s_idx),1)))
                    rng = np.random.RandomState(0)
                    take = np.concatenate([rng.choice(pos, min(npos, len(pos)), replace=False),
                                           rng.choice(neg, min(per_site - min(npos, len(pos)), len(neg)), replace=False)])
                    idx_pool.extend(take.tolist())
                # mark pool candidates as the training rows; eval unchanged
                # nested_cv_score trains on site membership -> emulate by temporarily scoring
                # with a custom mask: build a filtered row/x set where 'train_sites' = combo
                # and only pool candidates are kept as training (others of combo sites dropped)
                train_rows = [all_rows[i] for i in idx_pool]
                train_x = all_x[idx_pool]
                test_mask = site_of == D
                test_idx = np.nonzero(test_mask)[0]
                test_rows = [all_rows[i] for i in test_idx]
                tx, txx = train_x, all_x[test_idx]
                tfolds = np.array([r["fold"] for r in train_rows])
                typ = np.array([1 if r["touched"] else 0 for r in train_rows])
                tcases = {c: v for c, v in all_cases.items() if v["site"] in combo}
                d_cases = {c: v for c, v in all_cases.items() if v["site"] == D}
                fit_predict = fit_predict_features(model_kind)
                for seed in seeds:
                    nf = int(tfolds.max()) + 1
                    oof = np.full(len(train_rows), np.nan)
                    for j in range(nf):
                        it, ih = tfolds != j, tfolds == j
                        if ih.sum() == 0 or it.sum() < 5:
                            continue
                        oof[ih] = fit_predict(tx[it], typ[it], tx[ih], seed=seed + j)
                    valid = ~np.isnan(oof)
                    v_idx = list(np.nonzero(valid)[0])
                    v_rows = [train_rows[i] for i in v_idx]
                    v_cases = {c: v for c, v in tcases.items() if any(r["case"] == c for r in v_rows)}
                    threshold = float(max(THRESHOLDS, key=lambda t: lesion_scores(v_rows, oof[valid] >= t, v_cases)["f1"]))   # 06.11. fix: ">= t" was missing
                    scores = fit_predict(tx, typ, txx, seed=seed)
                    sc = lesion_scores(test_rows, scores >= threshold, d_cases)
                    sc["threshold"] = threshold; sc["seed"] = seed
                    all_results.append(sc)
                f1 = mean_ci(all_results)[0]
                print(f"  FIXED D={D} k={k} combo={combo} F1={f1:.3f}", flush=True)
            f1, ci, _ = mean_ci(all_results)
            out[D][str(k)] = dict(f1=f1, ci95=ci, n_combos=len(combos), n_seeds=len(seeds), N_fix=N_fix)
    # bootstrap slope of F1 over k (per held-out site and pooled), 95 % CI
    def slope_ci(vals, n_boot=2000, rng_seed=0):
        ks, fs = zip(*vals)
        if len(ks) < 2:
            return 0.0, [0.0, 0.0]
        rng = np.random.RandomState(rng_seed)
        slopes = []
        for _ in range(n_boot):
            b = [vals[i] for i in rng.choice(len(vals), size=len(vals), replace=True)]
            if len(set(x[0] for x in b)) < 2:
                continue
            ks_b, fs_b = zip(*b)
            slopes.append(np.polyfit(np.asarray(ks_b, float), np.asarray(fs_b, float), 1)[0])
        slopes = np.asarray(slopes)
        return float(np.mean(slopes)), [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
    per_site = {}
    pooled_pts = []
    for D in out:
        if not isinstance(out[D], dict) or not any(isinstance(v, dict) and "f1" in v for v in out[D].values()):
            continue
        pts = sorted((int(k), v["f1"]) for k, v in out[D].items() if isinstance(v, dict) and "f1" in v)
        m, ci = slope_ci(pts)
        per_site[D] = dict(slope_per_site=m, ci95=ci)
        pooled_pts.extend(pts)
    m, ci = slope_ci(sorted(pooled_pts))
    out["slope_test"] = dict(method="bootstrap over k (2000 draws)", per_site=per_site,
                              pooled=dict(slope_per_site=m, ci95=ci),
                              note="positive slope = fixed-size F1 rises with the NUMBER of distinct sites at constant pool size")
    out["meta"] = dict(model=model_kind, duration_s=round(time.time()-t0,1),
                       note="fixed-size curve: pool subsampled to N_fix = smallest single-site pool of the training sites")
    return out

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["3a", "3c", "3c_fixed", "all"], default="3a")
    p.add_argument("--model", default="forest")
    p.add_argument("--smoke", action="store_true", help="run with 2 cases only for testing")
    args = p.parse_args()
    
    print("Loading all sites...", flush=True)
    all_rows, all_x, all_cases, all_sites = load_all()
    print(f"Sites: {all_sites}", flush=True)
    for s in all_sites:
        n = sum(1 for r in all_rows if r["site"] == s)
        nc = sum(1 for c in all_cases.values() if c["site"] == s)
        print(f"  {s}: {n} cands, {nc} cases", flush=True)
    
    if args.smoke:
        print("\n=== SMOKE TEST ===", flush=True)
        t0 = time.time()
        D = all_sites[0]
        others = [s for s in all_sites if s != D]
        r = nested_cv_score(all_rows, all_x, all_cases, args.model, [0], set(others), D)
        print(f"Smoke LODO: {D} F1={mean_ci(r)[0]:.3f} ({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
        r2 = in_dataset_cv(all_rows, all_x, all_cases, args.model, [0], D)
        print(f"Smoke in-dataset: {D} F1={mean_ci(r2)[0]:.3f} ({time.time()-t0:.0f}s)", flush=True)
        print("SMOKE OK", flush=True)
        return
    
    if args.phase in ("3a", "all"):
        print("\n=== PHASE 3A ===", flush=True)
        res = run_phase3a(all_rows, all_x, all_cases, all_sites, args.model)
        res.pop("PLATZHALTER-wk3", None)
        out = os.path.join(MB, "paper/phase3a_summary.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Written {out}", flush=True)
    
    if args.phase in ("3c", "all"):
        print("\n=== PHASE 3C ===", flush=True)
        res = run_phase3c(all_rows, all_x, all_cases, all_sites, args.model)
        out = os.path.join(MB, "paper/phase3c_summary.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Written {out}", flush=True)
        # GPU budget check: if projection > 40 GPU-h, note the cut (CPU here, but log it)
        n_runs = res["meta"]["n_runs"]
        mean_s = res["meta"]["mean_run_s"]
        projected_h = n_runs * mean_s / 3600
        res["gpu_budget_check"] = dict(n_runs=n_runs, mean_run_s=mean_s,
                                        projected_total_h=round(projected_h, 2),
                                        within_40h=projected_h <= 40,
                                        note="stage-2 feature models run on CPU; GPU not used")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        
    if args.phase in ("3c_fixed", "all"):
        print("\n=== PHASE 3C FIXED ===", flush=True)
        res = run_phase3c_fixed(all_rows, all_x, all_cases, all_sites, args.model)
        out = os.path.join(MB, "paper/phase3c_fixed_summary.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Written {out}", flush=True)
        
        # Plot fig4 (both curves: growing size vs fixed size) -- only if fixed data exists
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fixed_path = os.path.join(MB, "paper/phase3c_fixed_summary.json")
            fixed = json.load(open(fixed_path)) if os.path.exists(fixed_path) else None
            fig, ax = plt.subplots(figsize=(9, 6))
            for D, ks in res.items():
                if not isinstance(ks, dict) or not any(isinstance(v, dict) for v in ks.values()):
                    continue
                k_vals = sorted([int(k) for k, v in ks.items() if isinstance(v, dict) and "f1" in v])
                if not k_vals:
                    continue
                f1_vals = [ks[str(k)]["f1"] for k in k_vals]
                ci_lo = [ks[str(k)]["ci95"][0] for k in k_vals]
                ci_hi = [ks[str(k)]["ci95"][1] for k in k_vals]
                ax.plot(k_vals, f1_vals, "o-", label=f"{D} (growing)")
                ax.fill_between(k_vals, ci_lo, ci_hi, alpha=0.12)
            if fixed:
                for D, ks in fixed.items():
                    if not isinstance(ks, dict):
                        continue
                    k_vals = sorted([int(k) for k, v in ks.items() if isinstance(v, dict) and "f1" in v])
                    if not k_vals:
                        continue
                    f1_vals = [ks[str(k)]["f1"] for k in k_vals]
                    ax.plot(k_vals, f1_vals, "--", alpha=0.6)
            ax.set_xlabel("k (number of training sites)")
            ax.set_ylabel("F1 on held-out site")
            ax.set_title("III-C: Site-count curve (solid=growing pool, dashed=fixed size)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = os.path.join(ROOT, "paper/fig4_site_count_curve.png")
            plt.savefig(fig_path, dpi=150)
            print(f"Written {fig_path}", flush=True)
        except Exception as e:
            print(f"Plot failed: {e}", flush=True)

if __name__ == "__main__":
    main()
