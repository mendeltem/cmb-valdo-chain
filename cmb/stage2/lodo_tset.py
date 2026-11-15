"""wk3 redo (06.11.2026): Phase III leave-one-dataset-out with the SPECIFIED stage-2 classifier tset-plain (the paper's
pooled stage-2 CNN), not the feature forest. The LODO logic is cmb.stage2.lodo unchanged (sites, twins F1, nested
thresholds, combinations, fixed-size pools, slope test); only two things are swapped:
  * the inputs: a cmb.stage2.trackset.Bundle (patches + side features), built exactly like cmb.stage2.pooled does it,
    instead of the 12 hand-made features;
  * the model: cmb.stage2.trackset.MODELS["tset-plain"] (same fit_predict(train_x, train_y, test_x, seed) contract).
III-B is not run: it needs the pooled stage 1 of wk2, which was dropped on 03.11.2026.

One held-out site per call, so every site is its own HPC job:
    python -m cmb.stage2.lodo_tset --phase 3a --site RSS --out /abs/path/phase3a_RSS.json [--smoke]
    python -m cmb.stage2.lodo_tset --merge --phase 3a --parts DIR --out /abs/path/phase3a_summary.json
--smoke: one fit with 2 epochs, 1 member on a tiny train/test slice (pipeline check only).
"""
import argparse, functools, glob, json, os, time
import numpy as np

from cmb.stage2 import lodo
from cmb.stage2 import pooled                       # importing fills pooled.TISSUE_CLASS (SynthSeg label -> tissue group)
from cmb.stage2 import trackset as _tset

MODEL = "tset-plain"
FITS = {"3a": 51, "3c": 102, "3c_fixed": 102}      # tset fits per held-out site (counted from the lodo loops: 3a = 3 seeds x
                                                   # (6 nested + 1 calibrated + 10 in-dataset); 3c = (4+6+4) combos x 6 + 1 x 3 seeds x 6)
FITS_CUT = {"3a": 51, "3c": 72, "3c_fixed": 72}    # with at most 3 combinations per k (spec: if the projection exceeds 40 GPU-h)
BUDGET_H = 40.0


def cap_combos(n):
    """Spec rule (GPU budget): at most n combinations per k, all for k = 4 (lodo.combos_for order, first n)."""
    orig = lodo.combos_for
    lodo.combos_for = lambda others, k, rng_seed=42: orig(others, k, rng_seed) if k == 4 else orig(others, k, rng_seed)[:n]


def load_all_bundle():
    """Same rows/cases/sites as lodo.load_all, but all_x = a Bundle over the concatenated patches (pooled.py recipe)."""
    rows, cases, blocks = [], {}, []
    for name in lodo.SITES:
        info = json.load(open(f"{lodo.SITES[name]}/candidates.json"))
        patches = np.load(f"{lodo.SITES[name]}/patches.npy", mmap_mode="r")
        for case_id, case in info["cases"].items():
            site = lodo.valdo_site(case_id) if name == "valdo" else name
            cases[case_id] = dict(fold=case["fold"], n_reference=case["n_reference"], site=site)
        for row in info["candidates"]:
            rows.append(dict(row, fold=cases[row["case"]]["fold"], site=cases[row["case"]]["site"], case=row["case"]))
        blocks.append(np.asarray(patches))
        print(f"  {name}: {len(info['candidates'])} candidates, {len(info['cases'])} cases", flush=True)
    patches_all = np.concatenate(blocks)
    missing = sum("centre_voxel" not in r for r in rows)
    assert not missing, f"{missing} candidates without centre_voxel"
    case_ids = {c: i for i, c in enumerate(sorted({r["case"] for r in rows}))}
    case_index = np.array([case_ids[r["case"]] for r in rows])
    centre_mm = np.array([r["centre_voxel"] for r in rows], np.float32) * _tset.VOXEL_MM_XYZ
    bundle = _tset.Bundle(patches_all, _tset.side_from_rows(rows, patches_all, pooled.TISSUE_CLASS), case_index, centre_mm, None)
    return rows, bundle, cases, sorted({r["site"] for r in rows})


def use_tset(epochs=None, members=None):
    kw = {} if epochs is None else dict(epochs=epochs, members=members)
    fit = functools.partial(_tset.MODELS[MODEL], **kw)
    lodo.fit_predict_features = lambda kind: fit         # every lodo function asks for its model through this name


def _parts(parts, phase):
    """Part files of exactly this phase: phase3c_<site>.json, NOT phase3c_fixed_<site>.json (06.11.: the plain glob
    phase3c_*.json matched both, so the fixed-size parts overwrote the growing-pool ones). Site names hold no "_"."""
    pre = f"phase{phase}_"
    return [f for f in glob.glob(f"{parts}/{pre}*.json") if "_" not in os.path.basename(f)[len(pre):-5]]


def merge(phase, parts, out):
    res = {}
    for f in sorted(_parts(parts, phase)):
        part = json.load(open(f))
        res.update({k: v for k, v in part.items() if k not in ("meta", "slope_test")})
    if phase == "3c_fixed":                                # the slope test needs all sites: same bootstrap as lodo.run_phase3c_fixed, on the merged curves
        pts_all, per_site = [], {}
        def slope_ci(vals, n_boot=2000, rs=0):
            r = np.random.RandomState(rs); sl = []
            for _ in range(n_boot):
                b = [vals[i] for i in r.choice(len(vals), size=len(vals), replace=True)]
                if len({x[0] for x in b}) < 2:
                    continue
                ks, fs = zip(*b); sl.append(np.polyfit(np.asarray(ks, float), np.asarray(fs, float), 1)[0])
            sl = np.asarray(sl); return float(np.mean(sl)), [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]
        for D, v in res.items():
            pts = sorted((int(k), x["f1"]) for k, x in v.items() if isinstance(x, dict) and "f1" in x)
            if len(pts) >= 2:
                m, ci = slope_ci(pts); per_site[D] = dict(slope_per_site=m, ci95=ci); pts_all.extend(pts)
        m, ci = slope_ci(sorted(pts_all))
        res["slope_test"] = dict(method="bootstrap over k (2000 draws)", per_site=per_site, pooled=dict(slope_per_site=m, ci95=ci),
                                 note="positive slope = fixed-size F1 rises with the NUMBER of distinct sites at constant pool size")
    res["meta"] = dict(model=MODEL, sites=sorted(res.keys() - {"slope_test"}), merged_from=sorted(os.path.basename(f) for f in _parts(parts, phase)),
                       note="tset-plain redo of wk3 (HPC, 06.11.2026); LODO logic of cmb.stage2.lodo incl. the 06.11. threshold fixes")
    json.dump(res, open(out, "w"), indent=2)
    print("merged", len(res) - 1, "entries ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["3a", "3c", "3c_fixed"], required=True)
    ap.add_argument("--site"); ap.add_argument("--out", required=True)
    ap.add_argument("--merge", action="store_true"); ap.add_argument("--parts")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="time 3 full-recipe fits, project the total, write --out and the cut flag")
    ap.add_argument("--cut-flag", default=None, help="file: if it exists, 3c and 3c_fixed use at most 3 combinations per k")
    a = ap.parse_args()
    assert os.path.isabs(a.out), "--out must be an absolute path (pooled.py lesson 05.11.)"
    if a.merge:
        return merge(a.phase, a.parts, a.out)
    t0 = time.time()
    rows, bundle, cases, sites = load_all_bundle()
    assert a.site in sites, (a.site, sites)
    if a.smoke:                                            # pipeline check: one tiny fit on the GPU, no result file of the study
        use_tset(epochs=2, members=1)
        y = np.array([1 if r["touched"] else 0 for r in rows]); idx = np.arange(len(rows))
        rng = np.random.RandomState(0); tr = rng.choice(idx, 200, replace=False); te = rng.choice(np.setdiff1d(idx, tr), 50, replace=False)
        p = lodo.fit_predict_features(MODEL)(bundle[tr], y[tr], bundle[te], seed=0)
        ok = p.shape == (50,) and np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
        json.dump(dict(smoke=True, ok=bool(ok), n_rows=len(rows), sites=sites, seconds=round(time.time() - t0, 1)), open(a.out, "w"))
        print("SMOKE", "OK" if ok else "FAILED", flush=True)
        raise SystemExit(0 if ok else 1)
    if a.pilot:                                            # spec: measure the first runs, project, cut if > 40 GPU-h
        use_tset(); fit = lodo.fit_predict_features(MODEL)
        y = np.array([1 if r["touched"] else 0 for r in rows]); rng = np.random.RandomState(0); idx = rng.permutation(len(rows))
        n_te = len(rows) // 5; secs = []
        for i in range(3):                                 # full-pool size: train 4/5, test 1/5 (the largest fits of the study)
            te, tr = idx[i * n_te:(i + 1) * n_te], np.concatenate([idx[:i * n_te], idx[(i + 1) * n_te:]])
            t = time.time(); fit(bundle[tr], y[tr], bundle[te], seed=i); secs.append(time.time() - t)
        per = float(np.mean(secs)); total = sum(FITS.values()) * 5 * per / 3600; cut = total > BUDGET_H
        total_cut = sum(FITS_CUT.values()) * 5 * per / 3600
        rep = dict(fit_seconds=[round(v, 1) for v in secs], mean_fit_s=round(per, 1), fits_full=sum(FITS.values()) * 5,
                   projected_gpu_h=round(total, 1), budget_h=BUDGET_H, cut=cut, projected_gpu_h_after_cut=round(total_cut, 1),
                   note="upper bound: every fit timed at full-pool size; most LODO fits train on fewer candidates")
        json.dump(rep, open(a.out, "w"), indent=2); print("PILOT", rep, flush=True)
        if cut and a.cut_flag:
            open(a.cut_flag, "w").write(json.dumps(rep))
        return
    use_tset()
    if a.cut_flag and os.path.exists(a.cut_flag) and a.phase in ("3c", "3c_fixed"):
        cap_combos(3); print("GPU budget cut active: at most 3 combinations per k (all for k = 4)", flush=True)
    run = {"3a": lodo.run_phase3a, "3c": lodo.run_phase3c, "3c_fixed": lodo.run_phase3c_fixed}[a.phase]
    res = run(rows, bundle, cases, sites, model_kind=MODEL, only=[a.site])
    res["meta"] = dict(res.get("meta", {}), model=MODEL, site=a.site, seconds=round(time.time() - t0, 1))
    json.dump(res, open(a.out, "w"), indent=2)
    print("written", a.out, flush=True)


if __name__ == "__main__":
    main()
