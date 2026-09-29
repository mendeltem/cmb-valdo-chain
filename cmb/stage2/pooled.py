"""One second-stage classifier for ALL cohorts  (user idea, 2026-09-21).

"The U-Nets may differ per cohort or image, but the classifier should always see the same kind of result."
Each cohort keeps its own first stage; the candidates of all cohorts are pooled, and ONE classifier is trained on the pool.
Why this could work where cohort-wise second stages did not (ARBEIT.md 5d, 5j): the pool is several times larger
(VALDO ~200 candidates, private T2* ~960, private SWI ~870), and the hand-made features are physical (mm, contrast,
probability), so they should mean the same thing in every cohort.

Per cohort these operating modes are compared with the SAME nested cross-validation as in cmb.stage2.classifier:
  probability        no second stage: threshold on the stage-1 probability
  own <model>        classifier trained on the candidates of this cohort only           (skipped with --skip-own)
  pooled <model>     classifier trained on the candidates of all cohorts
  pooled+tag <model> the same with a one-hot cohort tag as extra features (feature models only)
  probability+csf    threshold on the probability plus the HARD rule 'majority tissue label is CSF -> reject' (needs candidates
                     built with --keep-csf, which store the label). The hard rule helps on VALDO (+0.06) but would delete 65
                     detected lesions on the private SWI cohort -- hence the soft alternative:
  ...+tissue         the tissue class of the candidate (CSF, cortex, white matter, deep nuclei, infratentorial) as five extra
                     one-hot features, so the classifier learns per cohort how much a CSF location should count (--tissue)
  pooled+extra ...   the pool enlarged by TRAINING-ONLY candidates given with --extra, e.g. the candidates of more sensitive
                     first stages (blob loss, centre map) for the same cases. They are never scored; for fold k only extra
                     candidates of cases outside fold k are used, so nothing leaks.
Models: logistic, forest, boosting (ten hand-made features, CPU), cnn (small plain 3D CNN on the central 16 mm cube of the stored
patches -- no residual connections --, GPU), cnn-ts (the same CNN as the STUDENT of a teacher-student pair: it learns from the
out-of-fold predictions of the inner folds blended with the hard labels -- MicrobleedNet distils its discriminator, we never had;
cnn-ts03 / cnn-ts07 vary the blend from 0.3 to 0.7), cnn-vor (the same CNN whose encoder is first trained without labels to
fill in masked-out regions of the candidate patches, then fine-tuned; cnn-vor20 shortens the pre-training) and cnn2s (the same CNN at two scales: 16 mm cube at full and 28 mm cube at half resolution;
needs candidates built with --half 32 32 16; GPU).
Outer loop = the five folds (a classifier for fold k never sees a candidate of fold k, of any cohort); the acceptance
threshold is chosen per cohort on out-of-fold scores inside the training folds and applied unchanged to fold k.
Subjects scanned in two cohorts must share a fold: give --folds NAME=MANIFEST to take the fold of a cohort from a manifest
(private SWI: manifest_swi_mix.json puts the seven twins of T2* subjects into the T2* fold).
Every mode is compared pairwise with "probability" (bootstrap over cases, 5 000 draws).

    python -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd t2star=/private/stage2/run swi=/private/stage2/run \
        [--extra valdo=ergebnisse/stage2/a03-aniso-e60-gd-vblob ...] [--folds swi=/private/manifest_swi_mix.json] \
        [--models logistic forest boosting cnn] [--skip-own] [--json out.json]
Sums only -- safe for the private cohorts. The cnn model needs the GPU: queue it, never start it next to a training.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Callable, Dict, List

import numpy as np

from cmb.stage2.classifier import (FEATURE_NAMES, THRESHOLDS, features_of, fit_predict_cnn, fit_predict_cnn_pretrained,
                                   fit_predict_cnn_two_scale, fit_predict_features)

ROOT = "/home/uchralt/data/work/valdo-t2s"
CSF_LABELS = {4, 5, 14, 15, 24, 43, 44}
TISSUE_CLASS = {}                                          # SynthSeg label -> 0..4 (same grouping as the tissue input channels)
for _labels, _k in (((4, 5, 14, 15, 24, 43, 44), 0), ((3, 42, 17, 18, 53, 54), 1), ((2, 41), 2),
                    ((10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60), 3), ((7, 8, 16, 46, 47), 4)):
    for _l in _labels:
        TISSUE_CLASS[_l] = _k


def scores_of(rows: List[Dict], accepted: np.ndarray, n_reference: int, n_cases: int) -> Dict[str, float]:
    """Lesion-level scores with the semantics of metric.hits (TP = reference lesions touched by an accepted candidate)."""
    found = {(r["case"], lesion) for r, keep in zip(rows, accepted) if keep for lesion in r["touched"]}
    fp = sum(1 for r, keep in zip(rows, accepted) if keep and not r["touched"])
    tp = len(found); fn = n_reference - tp
    return dict(f1=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1), sensitivity=tp / max(tp + fn, 1),
                fp_per_case=fp / max(n_cases, 1))


def best_threshold(rows, scores, n_reference, n_cases) -> float:
    return float(max(THRESHOLDS, key=lambda t: scores_of(rows, scores >= t, n_reference, n_cases)["f1"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", nargs="+", required=True, help="NAME=FOLDER with candidates.json and patches.npy")
    parser.add_argument("--extra", nargs="*", default=[], help="NAME=FOLDER: training-only candidates for cohort NAME (may repeat a name)")
    parser.add_argument("--folds", nargs="*", default=[], help="NAME=MANIFEST: take the folds of this cohort from a manifest")
    parser.add_argument("--models", nargs="*", default=["logistic", "forest", "boosting"])
    parser.add_argument("--tissue", action="store_true", help="add the modes that use the stored tissue label (hard CSF rule, tissue features)")
    parser.add_argument("--skip-own", action="store_true", help="skip the cohort-wise classifiers (saves GPU time for cnn)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="verschiebt ALLE Startwerte der zweiten Stufe. 0 = wie bisher (bitgleich). Fuer eine Messreihe: denselben Befehl "
                             "mit 0..9 laufen lassen und die JSON-Dateien mit cmb.stage2.compare_repeats zusammenfassen -- der Fall-Bootstrap "
                             "enthaelt das Trainingsrauschen der zweiten Stufe (~0.03) nicht, nur Wiederholungen zeigen es.")
    parser.add_argument("--transfer", default=None, metavar="SOURCE=TARGET,TARGET",
                        help="arm 4 (2026-09-28): CNN pre-trained on SOURCE cohort candidates, fine-tuned on each TARGET's training folds")
    parser.add_argument("--fractions", type=float, nargs="*", default=[0.0, 0.1, 0.25, 0.5, 1.0],
                        help="--transfer: share of the target TRAINING CASES used for fine-tuning (0 = zero-shot classifier)")
    parser.add_argument("--transfer-epochs", type=int, nargs=2, default=[40, 20], metavar=("PRE", "FINE"))
    parser.add_argument("--json")
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    fold_override = {}
    for spec in args.folds:
        name, manifest = spec.split("=", 1)
        fold_override[name] = {e["id"]: e["fold"] for e in json.load(open(absolute(manifest)))}

    rows: List[Dict] = []; features: List[List[float]] = []; patch_blocks = []; cases: Dict[str, Dict] = {}
    names: List[str] = []
    import functools
    from cmb.stage2.classifier import fit_predict_cnn_transfer as _fit_transfer
    PATCH_MODELS = {"cnn": fit_predict_cnn, "cnn2s": fit_predict_cnn_two_scale,
                    # Arm 5 Schritt 3 (29.09.2026): derselbe CNN, aber 32 mm entlang z (16 mm in der Schicht) -- braucht Kandidaten mit --half 20 20 20
                    "cnn-z32": functools.partial(fit_predict_cnn, crop=(32, 32, 32)),
                    # selbstueberwachtes Vortraining des Encoders auf den EIGENEN Ausschnitten, ohne Etiketten
                    "cnn-vor": fit_predict_cnn_pretrained,
                    "cnn-vor20": functools.partial(fit_predict_cnn_pretrained, pretrain_epochs=20),
                    # Lehrer-Schueler: der Schueler bekommt die Vorhersagen der inneren Falten als weiche Ziele
                    "cnn-ts": fit_predict_cnn,
                    "cnn-ts03": functools.partial(fit_predict_cnn, alpha=0.3),
                    "cnn-ts07": functools.partial(fit_predict_cnn, alpha=0.7)}
    DESTILLIERT = {"cnn-ts", "cnn-ts03", "cnn-ts07"}
    need_patches = any(m in PATCH_MODELS for m in args.models) or bool(args.transfer)
    for spec, is_extra in [(s, False) for s in args.cohort] + [(s, True) for s in args.extra]:
        name, folder = spec.split("=", 1)
        if not is_extra:
            names.append(name)
        info = json.load(open(f"{absolute(folder)}/candidates.json"))
        patches = np.load(f"{absolute(folder)}/patches.npy", mmap_mode="r")
        moved = 0
        for case_id, case in info["cases"].items():
            fold = fold_override.get(name, {}).get(case_id, case["fold"]); moved += fold != case["fold"]
            if not is_extra:
                cases[case_id] = dict(fold=fold, n_reference=case["n_reference"], cohort=name)
            else:
                assert case_id in cases and cases[case_id]["fold"] == fold, f"extra candidates of an unknown case or fold: {case_id}"
        for i, row in enumerate(info["candidates"]):
            rows.append(dict(row, fold=cases[row["case"]]["fold"], cohort=name, extra=is_extra))
            features.append(features_of(np.asarray(patches[i]), row, info["threshold"]))
        if need_patches:
            patch_blocks.append(np.asarray(patches))
        print(f"{name}{' (extra, training only)' if is_extra else ''}: {len(info['candidates'])} candidates, "
              f"{sum(bool(r['touched']) for r in info['candidates'])} on a microbleed"
              + ("" if is_extra else f", {sum(c['n_reference'] for c in info['cases'].values())} reference lesions in {len(info['cases'])} cases")
              + (f", {moved} cases moved to the fold of their twin" if moved and not is_extra else ""), flush=True)
    x = np.asarray(features, np.float32)
    patches_all = np.concatenate(patch_blocks) if need_patches else None
    tag = np.stack([[float(r["cohort"] == n) for n in names] for r in rows]).astype(np.float32)
    tissue = np.zeros((len(rows), 5), np.float32)
    for i, r in enumerate(rows):
        if r.get("tissue", -1) in TISSUE_CLASS:
            tissue[i, TISSUE_CLASS[r["tissue"]]] = 1.0
    in_csf = tissue[:, 0].copy()
    y = np.array([1 if r["touched"] else 0 for r in rows]); fold = np.array([r["fold"] for r in rows])
    cohort = np.array([r["cohort"] for r in rows]); extra = np.array([r["extra"] for r in rows])

    def reference(name: str, keep_fold) -> tuple:
        here = [c for c in cases.values() if c["cohort"] == name and keep_fold(c["fold"])]
        return sum(c["n_reference"] for c in here), len(here)

    def nested(inputs: np.ndarray, fit_predict: Callable, pool_of: Callable[[str], np.ndarray], shared: bool,
               destillieren: bool = False) -> np.ndarray:
        """-> accepted flag per candidate (extra candidates stay False).
        pool_of(name) = candidates the classifier for cohort `name` may train on; shared=True means that pool is the same
        for every cohort, so one classifier per fold serves all cohorts (matters for the GPU model)."""
        accepted = np.zeros(len(rows), bool)
        groups = [names] if shared else [[n] for n in names]
        for group in groups:
            pool = pool_of(group[0])
            for k in range(5):
                train = np.nonzero(pool & (fold != k))[0]
                inner = np.zeros(len(train), np.float32)
                for j in sorted(set(fold[train])):
                    held = fold[train] == j
                    inner[held] = fit_predict(inputs[train[~held]], y[train[~held]], inputs[train[held]], seed=k * 10 + j + 1000 * args.seed_offset)
                test = np.nonzero(np.isin(cohort, group) & ~extra & (fold == k))[0]
                # Lehrer-Schueler: `inner` sind die Vorhersagen der INNEREN Falten auf genau diesen Trainingskandidaten,
                # also ehrliche Lehrer-Wahrscheinlichkeiten ausserhalb der jeweiligen Falte -- sie kosten nichts extra,
                # die geschachtelte Auswertung rechnet sie ohnehin fuer die Schwellenwahl.
                zusatz = dict(soft=inner) if destillieren else {}
                test_scores = fit_predict(inputs[train], y[train], inputs[test], seed=k + 1000 * args.seed_offset, **zusatz)
                for name in group:                           # threshold from THIS cohort's own (non-extra) training candidates
                    own = (cohort[train] == name) & ~extra[train]
                    threshold = best_threshold([rows[i] for i in train[own]], inner[own], *reference(name, lambda f: f != k))
                    mine = cohort[test] == name
                    accepted[test[mine]] = test_scores[mine] >= threshold
        return accepted

    own_pool = lambda name: (cohort == name) & ~extra
    all_pool = lambda name: ~extra
    all_plus_extra = lambda name: np.ones(len(rows), bool)
    modes = [("probability", x[:, :1], lambda tx, ty, sx, seed=0: sx[:, 0], own_pool, False, False)]
    if args.tissue:
        modes.append(("probability+csf", np.stack([x[:, 0], in_csf], 1), lambda tx, ty, sx, seed=0: sx[:, 0] * (1 - sx[:, 1]), own_pool, False, False))
    for model in args.models:
        inputs, fit = (patches_all, PATCH_MODELS[model]) if model in PATCH_MODELS else (x, fit_predict_features(model))
        ts = model in DESTILLIERT
        if not args.skip_own:
            modes.append((f"own {model}", inputs, fit, own_pool, False, ts))
        modes.append((f"pooled {model}", inputs, fit, all_pool, True, ts))
        if model not in PATCH_MODELS:
            modes.append((f"pooled+tag {model}", np.concatenate([x, tag], 1), fit, all_pool, True, ts))
        if args.tissue and model not in PATCH_MODELS:
            with_tissue = np.concatenate([x, tissue], 1)
            if not args.skip_own:
                modes.append((f"own+tissue {model}", with_tissue, fit, own_pool, False, ts))
            modes.append((f"pooled+tissue {model}", with_tissue, fit, all_pool, True, ts))
            modes.append((f"pooled+tissue+tag {model}", np.concatenate([x, tissue, tag], 1), fit, all_pool, True, ts))
            if args.extra:
                modes.append((f"pooled+tissue+extra {model}", with_tissue, fit, all_plus_extra, True, ts))
        elif args.extra:
            modes.append((f"pooled+extra {model}", inputs, fit, all_plus_extra, True, ts))

    def per_case(name: str, accepted: np.ndarray) -> Dict[str, tuple]:
        """case id -> (tp, fp, fn) for the paired comparison over cases."""
        counts = {c: [set(), 0] for c, v in cases.items() if v["cohort"] == name}
        for r, keep in zip(rows, accepted):
            if keep and r["cohort"] == name and not r["extra"]:
                if r["touched"]:
                    counts[r["case"]][0].update(r["touched"])
                else:
                    counts[r["case"]][1] += 1
        return {c: (len(found), fp, cases[c]["n_reference"] - len(found)) for c, (found, fp) in counts.items()}

    def pooled_f1(triples) -> float:
        tp, fp, fn = (sum(t[i] for t in triples) for i in range(3))
        return 2 * tp / max(2 * tp + fp + fn, 1)

    def reference_subset(name: str, chosen) -> tuple:
        here = [v for c, v in cases.items() if v["cohort"] == name and c in chosen]
        return sum(c["n_reference"] for c in here), len(here)

    def transfer_nested(source: str, target: str, fraction: float, cache: dict) -> np.ndarray:
        """Arm 4: pre-train on ALL non-extra candidates of `source` (they come from the source's own out-of-fold U-Net maps and
        have no relation to the target's folds), fine-tune on a case-level share of the target's training folds, threshold
        from the fine-tuning candidates via inner folds (share 0: threshold from the source's own inner folds)."""
        accepted = np.zeros(len(rows), bool)
        src = np.nonzero((cohort == source) & ~extra)[0]
        pre, fine = args.transfer_epochs
        fit = lambda ft_idx, test_idx, seed: _fit_transfer(patches_all[src], y[src], patches_all[ft_idx], y[ft_idx], patches_all[test_idx],
                                                            seed=seed, epochs=pre, finetune_epochs=fine, cache=cache)
        for k in range(5):
            seed = k + 1000 * args.seed_offset
            train_all = np.nonzero((cohort == target) & ~extra & (fold != k))[0]
            case_ids = sorted({rows[i]["case"] for i in train_all})
            order = np.random.RandomState(seed).permutation(len(case_ids))          # nested subsets across fractions
            chosen = {case_ids[j] for j in order[:int(round(fraction * len(case_ids)))]}
            train = np.array([i for i in train_all if rows[i]["case"] in chosen], int)
            if len(train):
                inner = np.zeros(len(train), np.float32)
                for j in sorted(set(fold[train])):
                    held = fold[train] == j
                    inner[held] = fit(train[~held], train[held], seed * 10 + j)
                threshold = best_threshold([rows[i] for i in train], inner, *reference_subset(target, chosen))
            else:                                                            # zero-shot: threshold from the source's inner folds
                key = ("src-inner", seed)
                if key not in cache:
                    inner_src = np.zeros(len(src), np.float32)
                    for j in sorted(set(fold[src])):
                        held = fold[src] == j
                        inner_src[held] = _fit_transfer(patches_all[src[~held]], y[src[~held]], patches_all[:0], y[:0], patches_all[src[held]],
                                                        seed=seed * 10 + j, epochs=pre, finetune_epochs=fine, cache=None)
                    cache[key] = best_threshold([rows[i] for i in src], inner_src, *reference(source, lambda f: True))
                threshold = cache[key]
            test = np.nonzero((cohort == target) & ~extra & (fold == k))[0]
            accepted[test] = fit(train, test, seed) >= threshold
            print(f"  transfer {source}->{target} share {fraction:g} fold {k}: {len(train)} fine-tuning candidates from {len(chosen)} cases, "
                  f"threshold {threshold:.3f}", flush=True)
        return accepted

    out = {}; kept = {}
    for label, inputs, fit_predict, pool_of, shared, destilliert in modes:
        accepted = nested(inputs, fit_predict, pool_of, shared, destilliert)
        kept[label] = accepted
        out[label] = {}
        for name in names:
            sel = (cohort == name) & ~extra
            out[label][name] = {k: round(v, 3) for k, v in scores_of([r for r, s in zip(rows, sel) if s], accepted[sel], *reference(name, lambda f: True)).items()}
        print(f"{label:30s} " + "   ".join(f"{n}: F1 {out[label][n]['f1']:.3f} P {out[label][n]['precision']:.2f} S {out[label][n]['sensitivity']:.2f} "
                                          f"FP {out[label][n]['fp_per_case']:.2f}" for n in names), flush=True)

    if args.transfer:
        source, targets = args.transfer.split("="); targets = targets.split(",")
        assert source in names and all(t in names for t in targets), args.transfer
        assert patches_all is not None, "--transfer needs a patch model (--models cnn)"
        cache = {}
        for fraction in args.fractions:
            label = f"transfer{fraction:g} cnn"
            accepted = np.zeros(len(rows), bool)
            for target in targets:
                accepted |= transfer_nested(source, target, fraction, cache)
            kept[label] = accepted
            out[label] = {}
            for name in names:
                sel = (cohort == name) & ~extra
                out[label][name] = {k: round(v, 3) for k, v in scores_of([r for r, s in zip(rows, sel) if s], accepted[sel], *reference(name, lambda f: True)).items()}
                if name not in targets:
                    out[label][name]["note"] = "source cohort: not scored in this mode"
            print(f"{label:30s} " + "   ".join(f"{n}: F1 {out[label][n]['f1']:.3f} P {out[label][n]['precision']:.2f} S {out[label][n]['sensitivity']:.2f} "
                                              f"FP {out[label][n]['fp_per_case']:.2f}" for n in targets), flush=True)

    rng = np.random.RandomState(0)
    for base in [b for b in ("probability", "probability+csf") if b in kept]:
      print(f"\npaired against '{base}' (delta F1 [95 % interval]):")
      for label in kept:
        if label in ("probability", "probability+csf"):
            continue
        parts = []
        for name in names:
            a, b = per_case(name, kept[label]), per_case(name, kept[base])
            ids = sorted(a); A = [a[i] for i in ids]; B = [b[i] for i in ids]
            draws = []
            for _ in range(5000):
                ix = rng.randint(len(ids), size=len(ids)); draws.append(pooled_f1([A[j] for j in ix]) - pooled_f1([B[j] for j in ix]))
            lo, hi = np.percentile(draws, [2.5, 97.5]); delta = pooled_f1(A) - pooled_f1(B)
            out[label][name]["paired_vs_" + base] = dict(delta_f1=round(delta, 3), ci95=[round(float(lo), 3), round(float(hi), 3)])
            parts.append(f"{name} {delta:+.3f} [{lo:+.3f}; {hi:+.3f}]")
        print(f"{label:30s} " + "   ".join(parts), flush=True)
    if args.json:                        # per-case counts allow paired comparisons ACROSS candidate sets (cmb.stage2.compare_runs); private ids stay in this file
        counts = {label: {name: per_case(name, kept[label]) for name in names} for label in kept}
        json.dump(dict(features=FEATURE_NAMES, cohorts=names, extra=args.extra, seed_offset=args.seed_offset,
                       results=out, per_case=counts), open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
