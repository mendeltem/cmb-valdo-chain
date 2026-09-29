"""Apply the VALDO-only second stage to the candidates of ANOTHER cohort (deployment mode; WORKLOG.md 5ae arm 4, 2026-09-29).

The pooled classifier (cmb.stage2.pooled) needs the private candidates at training time. Arm 4 showed that a classifier
trained on VALDO candidates alone transfers zero-shot (catalina-SWI 0.671 vs 0.672 pooled). This is the tool that does
exactly that outside a cross-validation: the CNN (cmb.stage2.classifier.fit_predict_cnn_transfer, no fine-tuning) is trained
on ALL VALDO candidates, its acceptance threshold is chosen on VALDO's own inner folds (out-of-fold scores of the VALDO
candidates), and both are applied unchanged to the target candidates. Three seeds x three members are averaged.
Reported on the target, paired over cases (bootstrap, 5 000 draws): probability threshold only, threshold + hard CSF rule,
and the VALDO classifier. Nothing of the target is used for any choice.

    python -m cmb.stage2.apply --source ergebnisse/stage2/a03-aniso-e60-gd-weich \
        --target NAME=<folder with candidates.json + patches.npy> [NAME=...] [--seeds 3] [--json out.json]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from cmb.stage2.classifier import fit_predict_cnn_transfer, THRESHOLDS
from cmb.stage2.pooled import scores_of, best_threshold

ROOT = "/home/uchralt/data/work/valdo-t2s"


def load(folder):
    folder = folder if os.path.isabs(folder) else f"{ROOT}/{folder}"
    info = json.load(open(f"{folder}/candidates.json"))
    patches = np.asarray(np.load(f"{folder}/patches.npy", mmap_mode="r"))
    rows = info["candidates"]; y = np.array([1 if r["touched"] else 0 for r in rows]); fold = np.array([r["fold"] for r in rows])
    n_ref = sum(c["n_reference"] for c in info["cases"].values()); n_cases = len(info["cases"])
    return info, patches, rows, y, fold, n_ref, n_cases


def per_case(rows, accepted, cases):
    counts = {c: [set(), 0] for c in cases}
    for r, keep in zip(rows, accepted):
        if keep:
            if r["touched"]:
                counts[r["case"]][0].update(r["touched"])
            else:
                counts[r["case"]][1] += 1
    return {c: (len(found), fp, cases[c]["n_reference"] - len(found)) for c, (found, fp) in counts.items()}


def pooled_f1(triples):
    tp, fp, fn = (sum(t[i] for t in triples) for i in range(3))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True); ap.add_argument("--target", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--json")
    a = ap.parse_args()
    s_info, s_patches, s_rows, s_y, s_fold, s_ref, s_cases = load(a.source)
    empty = s_patches[:0]
    # 1) threshold from the source's own inner folds (out-of-fold scores of the source candidates), averaged over seeds
    inner = np.zeros(len(s_rows), np.float32)
    for seed in range(a.seeds):
        for j in sorted(set(s_fold)):
            held = s_fold == j
            inner[held] += fit_predict_cnn_transfer(s_patches[~held], s_y[~held], empty, s_y[:0], s_patches[held],
                                                    seed=100 + seed * 10 + j, epochs=a.epochs, cache=None) / a.seeds
    threshold = best_threshold(s_rows, inner, s_ref, s_cases)
    src_oof = scores_of(s_rows, inner >= threshold, s_ref, s_cases)
    print(f"source {a.source}: {len(s_rows)} candidates, threshold {threshold:.2f}, out-of-fold F1 {src_oof['f1']:.3f} (P {src_oof['precision']:.2f} S {src_oof['sensitivity']:.2f})", flush=True)
    out = dict(source=a.source, threshold=threshold, source_out_of_fold=src_oof, seeds=a.seeds, targets={})
    # 2) train on ALL source candidates (cached per seed), apply to every target
    cache = {}
    for spec in a.target:
        name, folder = spec.split("=", 1)
        t_info, t_patches, t_rows, t_y, _, t_ref, t_cases = load(folder)
        scores = np.zeros(len(t_rows), np.float32)
        for seed in range(a.seeds):
            scores += fit_predict_cnn_transfer(s_patches, s_y, empty, s_y[:0], t_patches, seed=seed, epochs=a.epochs, cache=cache) / a.seeds
        prob = np.array([r["probability"] for r in t_rows]); csf = np.array([r.get("tissue", -1) in (4, 5, 14, 15, 24, 43, 44) for r in t_rows])
        vox = np.array([r["voxels"] for r in t_rows])
        fixed = (prob >= 0.3) & (vox >= 8)                     # the project's fixed cell: threshold 0.3, >= 2 mm3 (8 voxels of 0.25 mm3)
        modes = {"probability": fixed,
                 "probability+csf": fixed & ~csf,
                 "valdo classifier": scores >= threshold}
        res = {}
        for label, acc in modes.items():
            res[label] = {k: round(v, 3) for k, v in scores_of(t_rows, acc, t_ref, t_cases).items()}
        rng = np.random.RandomState(0); ids = sorted(t_info["cases"])
        for base in ("probability", "probability+csf"):
            A = per_case(t_rows, modes["valdo classifier"], t_info["cases"]); B = per_case(t_rows, modes[base], t_info["cases"])
            draws = []
            for _ in range(5000):
                ix = rng.randint(len(ids), size=len(ids)); draws.append(pooled_f1([A[ids[i]] for i in ix]) - pooled_f1([B[ids[i]] for i in ix]))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            res["valdo classifier"]["paired_vs_" + base] = dict(delta_f1=round(pooled_f1(list(A.values())) - pooled_f1(list(B.values())), 3), ci95=[round(float(lo), 3), round(float(hi), 3)])
        out["targets"][name] = dict(candidates=len(t_rows), cases=t_cases, reference=t_ref, results=res,
                                    accepted=[dict(case=r["case"], score=round(float(s), 3)) for r, s, k in zip(t_rows, scores, modes["valdo classifier"]) if k])
        print(f"\n{name}: {len(t_rows)} candidates, {t_ref} reference lesions, {t_cases} cases")
        for label, r in res.items():
            print(f"  {label:18s} F1 {r['f1']:.3f} P {r['precision']:.2f} S {r['sensitivity']:.2f} FP/case {r['fp_per_case']:.2f}"
                  + (f"   paired vs probability {r['paired_vs_probability']['delta_f1']:+.3f} {r['paired_vs_probability']['ci95']}, vs +csf {r['paired_vs_probability+csf']['delta_f1']:+.3f} {r['paired_vs_probability+csf']['ci95']}" if 'paired_vs_probability' in r else ""), flush=True)
    if a.json:
        json.dump(out, open(a.json if os.path.isabs(a.json) else f"{ROOT}/{a.json}", "w"), indent=1)


if __name__ == "__main__":
    main()
