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
TrackSet arms (2026-10-25, cmb.stage2.trackset): tset-plain (the small CNN rebuilt in the TrackSet training loop = the control),
tset-d4 (in-plane D4 group convolutions), tset-d4proto (+ prototype margin loss), tset-proto, tset-track (learned slice track),
tset-ctx (soft tissue context), tset-set (attention over all candidates of a scan), tset-full (all modules). tset-track and
tset-full want candidates with a deeper z extent (--half 20 20 20): slices outside the stored patch count as missing.
--neighbours adds five features from the other candidates of the same scan to the feature models (pooled+nb <model>),
a cheap CPU proxy for the set module. Both need 'centre_voxel' in candidates.json (candidates built after 2026-09-22).
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
                                   fit_predict_cnn_two_scale, fit_predict_features, fit_predict_ensemble)

from cmb.paths import ROOT
CSF_LABELS = {4, 5, 14, 15, 24, 43, 44}
TISSUE_CLASS = {}                                          # SynthSeg label -> 0..4 (same grouping as the tissue input channels)
CHANNEL_SETS = {"all": (0, 1, 2), "prob": (2,), "prob-frst": (1, 2)}   # wk2 mechanism arms: patch channels of the plain encoder
# Schleife R01 (2026-11-04): CPU feature-model arms that attack the remaining Stage-2 FPs
# without a GPU. name -> (inputs, fit). "plain" = base features x only; "neighbour" = base
# features x + the per-scan set/neighbour features of cmb.stage2.trackset (incl.
# collinear_between_15mm). See S/schleife/literatur_r1.md for the motivation (Cho 2026,
# Ghafoorian 2017, Sundaresan 2023, Al-Masni 2020).
FEATURE_MODELS_R01 = {
    "feat-ens":    ("plain",     fit_predict_ensemble()),          # mean of logistic+forest+boosting (Cho 2026 / VALDO ensemble)
    "feat-nb":     ("neighbour", fit_predict_features("forest")),  # forest + set/neighbour features (Ghafoorian 2017)
    "feat-vessel": ("neighbour", fit_predict_features("boosting")),# boosting + neighbour incl. collinearity (Sundaresan 2023)
}
# Schleife R02 (2026-11-04): CPU stage-2 precision arms on top of the R01 basis
# (feat-vessel = boosting + neighbours). All three add features only, no patches, CPU:
#   neighbour-atlas    x + neighbour features + 3 venous-atlas values (Bernier 2018,
#                      cmb.stage2.atlas_features; zeros when the candidate set has none)
#   ens-neighbour      ensemble of logistic+forest+boosting, EACH trained on x + neighbour
#                      features (R01 feat-ens saw only base features and was the weakest arm)
#   neighbour-tissue   x + neighbour features + 5 tissue one-hot (soft CSF class)
ATLAS_NAMES = ["atlas_vein_p", "atlas_vein_max3mm", "atlas_vein_dist_mm"]
FEATURE_MODELS_R02 = {
    "feat-vessel-atlas": ("neighbour-atlas",  fit_predict_features("boosting")),
    "feat-ens-nb":       ("ens-neighbour",    fit_predict_ensemble()),
    "feat-vessel-tissue":("neighbour-tissue", fit_predict_features("boosting")),
}
FEATURE_MODELS = {**FEATURE_MODELS_R01, **FEATURE_MODELS_R02}
# Schleife R03 (2026-11-04): CPU stage-2 precision arms ON TOP OF the R02 best candidate
# (feat-vessel-tissue = boosting + neighbours + 5 tissue one-hot, 'carry to confirmation'
# nach Regeln v2). Die Schwächere Seite der Kette bleibt die verbleibende Präzision
# (swi/t2star FP/Fall 2.2-2.6 trotz tissue-Feature), also bauen alle drei R03-Arme
# weiter auf der tissue-Basis und ergänzen:
#   neighbour-tissue-atlas  x + neighbour + tissue + 3 venous-atlas values (R02 zeigte
#                          atlas +0.028 und tissue +0.028 getrennt, die Kombination nie)
#   ens-neighbour-tissue    ENSEMBLE (logistic+forest+boosting, Mittel der Scores),
#                          JEDER auf x + neighbour + tissue (R02 feat-ens-nb sah nur
#                          base+neighbour; R01 feat-ens nur base -> beide ohne tissue)
#   mine-neighbour-tissue   BOOSTING mit foldweisem, leakage-freiem Hard-Negative-Mining:
#                          der INNEREN Pass (Falten j!=k) findet auf den Trainings-
#                          kandidaten die eigenen FP; die werden als gewichtete Negatives
#                          (Gewicht 3) in den finalen Fit aufgenommen (Hassanzadeh: geminte
#                          FP als Trainings-Negative; AcrSeek 2026: Mining nur mit
#                          Leakage-Kontrolle -> deshalb strikt ausserhalb der Testfalte).
FEATURE_MODELS_R03 = {
    "feat-vessel-tissue-atlas": ("neighbour-tissue-atlas", fit_predict_features("boosting")),
    "feat-ens-nb-tissue":       ("ens-neighbour-tissue",   fit_predict_ensemble()),
}
# Schleife R04 (2026-11-04): CPU stage-2 precision arms ON TOP OF the R03 best candidate
# (feat-vessel-tissue-atlas = boosting + neighbours + tissue + 3 venous-atlas values,
# 'carry to confirmation', all 0.6987). The weak side stays stage-2 precision
# (swi/t2star FP/Fall ~2.0/2.45), so all R04 arms build on the tissue-atlas basis:
#   ens-neighbour-tissue-atlas  ENSEMBLE (logistic+forest+boosting, mean of scores),
#                          EACH member on x + neighbour + tissue + 3 venous-atlas values
#                          (R03: boosting-tissue-atlas 0.6987 vs ensemble nb-tissue 0.6932;
#                          the ensemble has never seen the atlas values). See
#                          S/schleife/literatur_r4.md [1][5] -- no direct literature gain,
#                          so the expectation is a corridor, not a guaranteed win.
#   mine-neighbour-tissue-atlas BOOSTING with fold-wise leakage-free hard-negative
#                          mining (same mine_nested as feat-mine-nb-tissue, weight 3)
#                          on the tissue-ATLAS basis (R03 mine without atlas: drop, but
#                          best swi FP/Fall 1.61 -- the mined FP types may hit harder when
#                          the atlas values are present).
#   +syn (not a FEATURE_MODELS entry): pooled+extra <model> with --extra = the wc6
#                          synthetic candidate sets (inserted bleeds/veins, same cases
#                          and folds, training-only by construction). The extra vein/negative
#                          candidates are the mimic negatives; inserted bleed candidates
#                          that touch a reference lesion are true positives. Caution: the
#                          wc6 realism gate said 'too few candidates' (B_veins 0-2), so
#                          the synthetic veins are not realism-confirmed.
FEATURE_MODELS_R04 = {
    "feat-ens-tissue-atlas": ("ens-neighbour-tissue-atlas", fit_predict_ensemble()),
}
# feat-mine-nb-tissue / feat-mine-tissue-atlas brauchen rows/y und werden in main()
# als Closure registriert (unten).
FEATURE_MODELS = {**FEATURE_MODELS, **FEATURE_MODELS_R03, **FEATURE_MODELS_R04}
def _atlas_columns(rows) -> np.ndarray:
    """(n,3) venous-atlas values at the candidate centre; zeros where the candidate set was
    not built with cmb.stage2.atlas_features (the arm then degrades to feat-vessel, never crashes)."""
    out = np.zeros((len(rows), len(ATLAS_NAMES)), np.float32)
    for i, r in enumerate(rows):
        for k, f in enumerate(ATLAS_NAMES):
            v = r.get(f)
            if v is not None:
                out[i, k] = v
    return out
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


def write_decisions(path, rows, cohort, fold, extra, kept, dumps, seed_offset) -> int:
    """One JSON line per (mode, non-extra candidate); the metrics recomputed from it equal scores_of (unit test)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for label, accepted in kept.items():
            scores, thr = dumps.get(label, (np.full(len(rows), np.nan), np.full(len(rows), np.nan)))
            for i, r in enumerate(rows):
                if extra[i]:
                    continue
                f.write(json.dumps(dict(mode=label, seed_offset=seed_offset, cohort=str(cohort[i]), case=r["case"],
                                        fold=int(fold[i]), accepted=bool(accepted[i]),
                                        score=None if np.isnan(scores[i]) else round(float(scores[i]), 5),
                                        threshold=None if np.isnan(thr[i]) else round(float(thr[i]), 5),
                                        touched=list(r["touched"]), centre_voxel=r.get("centre_voxel"),
                                        voxels=r.get("voxels"), tissue=r.get("tissue"),
                                        probability=r.get("probability"))) + "\n")
                n += 1
    return n


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
    parser.add_argument("--neighbours", action="store_true",
                        help="feature models also as 'pooled+nb <model>': + five features from the other candidates of the same scan "
                             "(cmb.stage2.trackset.neighbour_features; 2026-10-25)")
    parser.add_argument("--channels", default="prob,prob-frst",
                        help="wk2 mechanism arms (2026-10-29): comma list of patch-channel sets for tset-plain, each as a "
                             "comma-free tag. 'all' = image+FRST+prob (bit-identical to the plain tset-plain), "
                             "'prob' = stage-1 probability only (C-prob), 'prob-frst' = FRST + probability (C-pf), "
                             "or any index set like '0-2','1-2','2'. Default runs only the two reduced sets; C-all is the "
                             "existing 'own/pooled tset-plain' run (no extra compute).")
    parser.add_argument("--match-size-to", default=None, metavar="D",
                        help="wk2 size-matched control (2026-10-29): for each cohort a 'matched <model> D' mode whose training "
                             "pool is a random draw from the pooled non-extra training candidates (in proportion to cohort "
                             "size) with EXACTLY the same number of positive and negative candidates as D's own pool, per fold. "
                             "Evaluated on D only. Seed offsets 0,1,2 = draw AND initialisation. Old runs are untouched.")
    parser.add_argument("--pooled-finetune", action="store_true",
                        help="wk2b: add the P->FT arm 'pooled-ft <model>' -- pre-train on the pooled training candidates "
                             "of ALL cohorts (fold k of every cohort held out together), then fine-tune on the target "
                             "cohort's own training candidates with half the epochs and lr/2; threshold from the "
                             "target cohort's inner folds of the fine-tuning set (as in S/S)")
    parser.add_argument("--ft-epochs", type=int, nargs=2, default=[40, 20], metavar=("PRE", "FINE"),
                        help="wk2b P->FT epochs: pre-train budget and fine-tuning budget (default 40 20; the fine "
                             "learning rate is always half the pre lr of 1e-3)")
    parser.add_argument("--json")
    parser.add_argument("--dump-inputs", default=None, metavar="FILE.npz",
                        help="2026-11-04 Arena 2 HPC search: write the feature table of the FIRST 'pooled feat-*' mode (inputs, "
                             "labels, folds, cohorts, case ids, touched lesions, reference counts, column groups) and exit "
                             "without fitting. No patches or images are written.")
    parser.add_argument("--dump-decisions", default=None, metavar="FILE",
                        help="wk5 (31.10.2026): write one JSON line per (mode, non-extra candidate): cohort, case, fold, "
                             "out-of-fold stage-2 score, threshold, accepted, touched reference lesions, centre voxel, "
                             "voxels, tissue. Private (case ids); only recorded, nothing is computed differently.")
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
    from cmb.stage2 import trackset as _tset                  # TrackSet arms (2026-10-25); looked up at run time so tests can swap MODELS
    unknown = [m for m in args.models if m not in PATCH_MODELS and m not in _tset.MODELS and m not in FEATURE_MODELS
               and m not in ("logistic", "forest", "boosting", "feat-mine-nb-tissue", "feat-mine-tissue-atlas")]
    if unknown:
        raise SystemExit(f"unknown --models {unknown}")
    need_patches = any(m in PATCH_MODELS or m in _tset.MODELS for m in args.models) or bool(args.transfer)
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
    tset_models = [m for m in args.models if m in _tset.MODELS]
    r01_neighbours = [m for m in args.models if (m in FEATURE_MODELS and FEATURE_MODELS[m][0] != "plain")
                      or m in ("feat-mine-nb-tissue", "feat-mine-tissue-atlas")]
    bundle = None; neighbours = None
    if tset_models or args.neighbours or r01_neighbours:
        missing = sum("centre_voxel" not in r for r in rows)
        if missing:
            raise SystemExit(f"{missing} candidates without 'centre_voxel': rebuild them with the current cmb.stage2.candidates")
        case_ids = {c: i for i, c in enumerate(sorted({r["case"] for r in rows}))}
        case_index = np.array([case_ids[r["case"]] for r in rows])
        centre_mm = np.array([r["centre_voxel"] for r in rows], np.float32) * _tset.VOXEL_MM_XYZ
        if args.neighbours or r01_neighbours:
            neighbours = _tset.neighbour_features(case_index, centre_mm, x[:, 0])
        if tset_models:
            tracks = None
            if any("track" in _tset.ARMS.get(m, ()) for m in tset_models):
                import time
                t0 = time.time()
                tracks = _tset.track_candidates(patches_all)
                print(f"slice tracks for {len(rows)} candidates in {time.time() - t0:.0f} s "
                      f"(patch depth {patches_all.shape[2]} slices, {2 * _tset.TRACK_HALF + 1} tracked)", flush=True)
            bundle = _tset.Bundle(patches_all, _tset.side_from_rows(rows, patches_all, TISSUE_CLASS), case_index, centre_mm, tracks)

    def reference(name: str, keep_fold) -> tuple:
        here = [c for c in cases.values() if c["cohort"] == name and keep_fold(c["fold"])]
        return sum(c["n_reference"] for c in here), len(here)

    _dump = {}
    def nested(inputs: np.ndarray, fit_predict: Callable, pool_of: Callable[[str], np.ndarray], shared: bool,
               destillieren: bool = False) -> np.ndarray:
        """-> accepted flag per candidate (extra candidates stay False).
        pool_of(name) = candidates the classifier for cohort `name` may train on; shared=True means that pool is the same
        for every cohort, so one classifier per fold serves all cohorts (matters for the GPU model)."""
        accepted = np.zeros(len(rows), bool)
        _dump["scores"] = np.full(len(rows), np.nan); _dump["thr"] = np.full(len(rows), np.nan)
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
                    _dump["scores"][test[mine]] = test_scores[mine]; _dump["thr"][test[mine]] = threshold
        return accepted

    def mine_nested(inputs: np.ndarray, fit_predict: Callable, pool_of: Callable[[str], np.ndarray], shared: bool,
                    hard_weight: float = 3.0) -> np.ndarray:
        """R03 hard-negative mining (Hassanzadeh: YOLO-mined FPs as training negatives; AcrSeek 2026:
        leakage control per fold): the INNER pass (fold j != k within the training folds) yields honest
        out-of-fold scores for the training candidates of fold k; the ones it ACCEPTED are the model's own
        false positives (label 0 by construction) and enter the final fit with weight `hard_weight` on top
        of the plain training set. Test candidates of fold k are never seen in any form. The final model is
        the SAME boosting as feat-vessel-tissue, so the arm differs only in the weighted negative set.
        """
        accepted = np.zeros(len(rows), bool)
        _dump["scores"] = np.full(len(rows), np.nan); _dump["thr"] = np.full(len(rows), np.nan)
        groups = [names] if shared else [[n] for n in names]
        for group in groups:
            pool = pool_of(group[0])
            for k in range(5):
                train = np.nonzero(pool & (fold != k))[0]
                inner = np.zeros(len(train), np.float32)
                for j in sorted(set(fold[train])):
                    held = fold[train] == j
                    inner[held] = fit_predict(inputs[train[~held]], y[train[~held]], inputs[train[held]], seed=k * 10 + j + 1000 * args.seed_offset)
                fp_idx = train[(inner >= np.median(inner)) & (y[train] == 0)]   # own FP: accepted by the inner pass
                w = np.ones(len(train), np.float32); w[np.isin(train, fp_idx)] = hard_weight
                from sklearn.ensemble import HistGradientBoostingClassifier as _H
                model = _H(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                           class_weight="balanced", random_state=k + 1000 * args.seed_offset)
                model.fit(inputs[train], y[train], sample_weight=w)
                test = np.nonzero(np.isin(cohort, group) & ~extra & (fold == k))[0]
                test_scores = model.predict_proba(inputs[test])[:, 1]
                for name in group:
                    own = (cohort[train] == name) & ~extra[train]
                    threshold = best_threshold([rows[i] for i in train[own]], inner[own], *reference(name, lambda f: f != k))
                    here = cohort[test] == name
                    accepted[test[here]] = test_scores[here] >= threshold
                    _dump["scores"][test[here]] = test_scores[here]; _dump["thr"][test[here]] = threshold
        return accepted

    own_pool = lambda name: (cohort == name) & ~extra
    all_pool = lambda name: ~extra
    all_plus_extra = lambda name: np.ones(len(rows), bool)
    modes = [("probability", x[:, :1], lambda tx, ty, sx, seed=0: sx[:, 0], own_pool, False, False)]
    if args.tissue:
        modes.append(("probability+csf", np.stack([x[:, 0], in_csf], 1), lambda tx, ty, sx, seed=0: sx[:, 0] * (1 - sx[:, 1]), own_pool, False, False))
    for model in args.models:
        if model in FEATURE_MODELS:
            # CPU feature-model arms (Schleife R01/R02): 'plain' uses the base features x; the
            # others append per-scan set/neighbour features and/or the venous-atlas values and/or
            # the tissue one-hot (all already computed above). Both pooled and own are registered,
            # mirroring the standard feature models below.
            kind, fit = FEATURE_MODELS[model]
            if kind == "plain":
                inputs = x
            elif kind == "neighbour":
                inputs = np.concatenate([x, neighbours], 1)
            elif kind == "neighbour-atlas":
                inputs = np.concatenate([x, neighbours, _atlas_columns(rows)], 1)
            elif kind == "neighbour-tissue":
                inputs = np.concatenate([x, neighbours, tissue], 1)
            elif kind == "neighbour-tissue-atlas":
                inputs = np.concatenate([x, neighbours, tissue, _atlas_columns(rows)], 1)
            elif kind == "ens-neighbour-tissue":
                inputs = np.concatenate([x, neighbours, tissue], 1)
            elif kind == "ens-neighbour-tissue-atlas":
                inputs = np.concatenate([x, neighbours, tissue, _atlas_columns(rows)], 1)
            else:
                inputs = np.concatenate([x, neighbours], 1)
            if not args.skip_own:
                modes.append((f"own {model}", inputs, fit, own_pool, False, False))
            modes.append((f"pooled {model}", inputs, fit, all_pool, True, False))
            if args.extra:
                # R04: training-only extra candidates (e.g. the wc6 synthetic mimic
                # sets) enlarge the pool for the feature models too. They are never
                # scored: nested() keeps extra rows out of `test` and scores_of skips them.
                modes.append((f"pooled+extra {model}", inputs, fit, all_plus_extra, True, False))
            continue
        if model in ("feat-mine-nb-tissue", "feat-mine-tissue-atlas"):
            # R03/R04 hard-negative-mining arm: the final per-fold model is trained with the
            # inner pass's own FPs as weighted negatives (mine_nested, leakage-free by fold
            # construction). fit=None is handled in the mode loop below. R03 (feat-mine-nb-
            # tissue) uses the tissue basis; R04 (feat-mine-tissue-atlas) uses the R03
            # best-candidate basis tissue + 3 venous-atlas values (mining on the stronger model).
            inputs = (np.concatenate([x, neighbours, tissue, _atlas_columns(rows)], 1)
                      if model == "feat-mine-tissue-atlas" else np.concatenate([x, neighbours, tissue], 1))
            modes.append((f"pooled {model}", inputs, None, all_pool, True, False))
            continue
        if model in _tset.MODELS:
            fit = functools.partial(lambda *a, _m=model, **k: _tset.MODELS[_m](*a, **k))
            if not args.skip_own:
                modes.append((f"own {model}", bundle, fit, own_pool, False, False))
            modes.append((f"pooled {model}", bundle, fit, all_pool, True, False))
            if args.extra:
                modes.append((f"pooled+extra {model}", bundle, fit, all_plus_extra, True, False))
            if "tset-plain" in model:
                # wk2 mechanism arms: reduced patch-channel sets of the plain encoder (C-prob / C-pf),
                # both own and pooled; 'all' is the plain run itself (bit-identical, never repeated here)
                for tag in [t for t in args.channels.split(",") if t != "all"]:
                    ch = CHANNEL_SETS[tag]
                    fit_ch = functools.partial(lambda *a, _m=model, **k: _tset.MODELS[_m](*a, channels=ch, **k))
                    if not args.skip_own:
                        modes.append((f"own {model} -{tag}", bundle, fit_ch, own_pool, False, False))
                    modes.append((f"pooled {model} -{tag}", bundle, fit_ch, all_pool, True, False))
            continue
        inputs, fit = (patches_all, PATCH_MODELS[model]) if model in PATCH_MODELS else (x, fit_predict_features(model))
        ts = model in DESTILLIERT
        if neighbours is not None and model not in PATCH_MODELS:
            modes.append((f"pooled+nb {model}", np.concatenate([x, neighbours], 1), fit, all_pool, True, ts))
            if args.tissue:
                modes.append((f"pooled+tissue+nb {model}", np.concatenate([x, tissue, neighbours], 1), fit, all_pool, True, ts))
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

    out = {}; kept = {}; dumps = {}
    if args.dump_inputs:
        label, inputs = next((m[0], m[1]) for m in modes if m[0].startswith("pooled feat-") and isinstance(m[1], np.ndarray))
        widths = {"base": x.shape[1], "neighbour": 0 if neighbours is None else neighbours.shape[1], "tissue": 5, "atlas": len(ATLAS_NAMES)}
        np.savez_compressed(args.dump_inputs, inputs=inputs.astype(np.float32), y=y, fold=fold, cohort=cohort, extra=extra,
                            case=np.array([r["case"] for r in rows]), touched=np.array([json.dumps(list(r["touched"])) for r in rows]),
                            cases=json.dumps(cases), names=np.array(names), thresholds=np.asarray(THRESHOLDS),
                            meta=json.dumps(dict(mode=label, columns=inputs.shape[1], widths=widths, models=args.models,
                                                 seed_offset=args.seed_offset, cohorts=args.cohort, folds=args.folds)))
        print(f"dumped {label}: {inputs.shape[0]} candidates x {inputs.shape[1]} columns -> {args.dump_inputs}", flush=True)
        return
    for label, inputs, fit_predict, pool_of, shared, destilliert in modes:
        if fit_predict is None:
            assert "mine" in label
            accepted = mine_nested(inputs, fit_predict_features("boosting"), pool_of, shared)
        else:
            accepted = nested(inputs, fit_predict, pool_of, shared, destilliert)
        kept[label] = accepted
        if args.dump_decisions:
            dumps[label] = (_dump["scores"].copy(), _dump["thr"].copy())
        out[label] = {}
        for name in names:
            sel = (cohort == name) & ~extra
            out[label][name] = {k: round(v, 3) for k, v in scores_of([r for r, s in zip(rows, sel) if s], accepted[sel], *reference(name, lambda f: True)).items()}
        print(f"{label:30s} " + "   ".join(f"{n}: F1 {out[label][n]['f1']:.3f} P {out[label][n]['precision']:.2f} S {out[label][n]['sensitivity']:.2f} "
                                          f"FP {out[label][n]['fp_per_case']:.2f}" for n in names), flush=True)

    if args.match_size_to:
        # wk2 size-matched control (user 30.10.2026, reviewer "more data or more diverse data?"): for each target
        # cohort D a mixed classifier trained on EXACTLY as many training candidates as D's separate classifier sees
        # in the same fold (same number of true and false), drawn at random from the pooled non-extra training
        # candidates of ALL cohorts (sampling without replacement from the union is automatically in proportion to
        # cohort size). Fold k of every cohort is held out together, so no test-fold case is ever drawn.
        # Seed offset 0,1,2 = draw AND initialisation.
        for model in [m for m in args.models if m in _tset.MODELS]:
            fit = functools.partial(lambda *a, _m=model, **k: _tset.MODELS[_m](*a, **k))
            for target in args.match_size_to.split(","):
                assert target in names, f"--match-size-to: unknown cohort {target}"
                label = f"matched {model} {target}"
                accepted = np.zeros(len(rows), bool)
                for k in range(5):
                    own_idx = np.nonzero((cohort == target) & ~extra & (fold != k))[0]
                    n_pos = int(y[own_idx].sum()); n_neg = len(own_idx) - n_pos
                    pool_pos = np.nonzero(~extra & (fold != k) & (y == 1))[0]
                    pool_neg = np.nonzero(~extra & (fold != k) & (y == 0))[0]
                    rng = np.random.RandomState(1000 * args.seed_offset + 100 + k)
                    draw = np.concatenate([rng.choice(pool_pos, n_pos, replace=False), rng.choice(pool_neg, n_neg, replace=False)])
                    d_y, d_fold = y[draw], fold[draw]
                    inner = np.zeros(len(draw), np.float32)
                    for j in sorted(set(d_fold)):
                        held = d_fold == j
                        inner[held] = fit(bundle[draw[~held]], y[draw[~held]], bundle[draw[held]], seed=1000 * args.seed_offset + 10 + j)
                    draw_cases = sorted({rows[i]["case"] for i in draw})
                    n_ref = sum(cases[c]["n_reference"] for c in draw_cases)
                    threshold = best_threshold([rows[i] for i in draw], inner, n_ref, len(draw_cases))
                    test = np.nonzero((cohort == target) & ~extra & (fold == k))[0]
                    accepted[test] = fit(bundle[draw], d_y, bundle[test], seed=k + 1000 * args.seed_offset) >= threshold
                    print(f"  matched {model} {target} fold {k}: draw {len(draw)} candidates "
                          f"({n_pos} pos / {n_neg} neg), threshold {threshold:.3f}", flush=True)
                kept[label] = accepted
                out[label] = {}
                for name in names:
                    sel = (cohort == name) & ~extra
                    if name == target:
                        out[label][name] = {k2: round(v, 3) for k2, v in scores_of([r for r, s in zip(rows, sel) if s], accepted[sel], *reference(name, lambda f: True)).items()}
                    else:
                        out[label][name] = {"note": "not evaluated (size-matched control is per dataset)"}
                print(f"{label:30s} {target}: F1 {out[label][target]['f1']:.3f} P {out[label][target]['precision']:.2f} "
                      f"S {out[label][target]['sensitivity']:.2f} FP {out[label][target]['fp_per_case']:.2f}", flush=True)

    if args.pooled_finetune:
        # wk2b (user 04.11.2026: "the second stage classifier can be trained with all the data or fine-tuned
        # with all the data"): arm P->FT, per target cohort D and test fold k (ARBEIT.md ## wk2b, written before
        # the first run):
        #   1. PRE-TRAIN the same tset-plain model (same patches, channels 'all') on the pooled training
        #      candidates of ALL cohorts with fold != k (fold k of every cohort held out together, exactly
        #      like S/P) -- full recipe: 40 epochs, lr 1e-3;
        #   2. FINE-TUNE on D's own training candidates (fold != k) with HALF the epochs (20) and lr/2 (5e-4);
        #   3. threshold from D's inner folds of the fine-tuning set (as in S/S); predict D's fold k.
        # The inner fits are equally honest: their pre-training set holds out fold j of every cohort as well,
        # so the threshold never sees an accepted-by-memory candidate. One case is predicted exactly once,
        # in its test fold.
        for model in [m for m in args.models if m in _tset.MODELS]:
            fit = functools.partial(lambda *a, _m=model, **k: _tset.MODELS[_m](*a, **k))
            label = f"pooled-ft {model}"
            pre_ep, fine_ep = args.ft_epochs
            fine_lr = 5e-4                                              # half of the pre-train lr of 1e-3 (fit_predict_trackset)
            accepted = np.zeros(len(rows), bool)
            pretrain_cache: dict = {}
            for target in names:
                fine_mask = (cohort == target) & ~extra
                for k in range(5):
                    fine = np.nonzero(fine_mask & (fold != k))[0]       # D's own training candidates (fold != k)
                    pre = np.nonzero(~extra & (fold != k))[0]           # pooled training candidates, ALL cohorts, fold != k
                    fine_folds = sorted(set(fold[fine]))
                    inner = np.zeros(len(fine), np.float32)
                    for j in fine_folds:                                # honest inner scores for D's threshold:
                        held = fold[fine] == j
                        fine_rest = fine[~held]
                        pre_j = np.nonzero(~extra & (fold != k) & (fold != j))[0]
                        inner[held] = fit(bundle[fine_rest], y[fine_rest], bundle[fine[held]],
                                          seed=k * 100 + j + 50 + 1000 * args.seed_offset,
                                          pretrain_x=bundle[pre_j], pretrain_y=y[pre_j],
                                          finetune_epochs=fine_ep, finetune_lr=fine_lr, pretrain_cache=pretrain_cache)
                    threshold = best_threshold([rows[i] for i in fine], inner, *reference(target, lambda f: f != k))
                    test = np.nonzero(fine_mask & (fold == k))[0]
                    accepted[test] = fit(bundle[fine], y[fine], bundle[test],
                                         seed=k + 1000 * names.index(target) + 1000 * args.seed_offset,
                                         pretrain_x=bundle[pre], pretrain_y=y[pre],
                                         finetune_epochs=fine_ep, finetune_lr=fine_lr, pretrain_cache=pretrain_cache) >= threshold
                    print(f"  pooled-ft {model} {target} fold {k}: pre-train {len(pre)} pooled candidates "
                          f"({pre_ep} ep), fine-tune {len(fine)} own ({fine_ep} ep, lr {fine_lr}), "
                          f"threshold {threshold:.3f}", flush=True)
            kept[label] = accepted
            out[label] = {}
            for name in names:
                sel = (cohort == name) & ~extra
                out[label][name] = {k2: round(v, 3) for k2, v in scores_of([r for r, s in zip(rows, sel) if s],
                                                                           accepted[sel], *reference(name, lambda f: True)).items()}
            print(f"{label:30s} " + "   ".join(f"{n}: F1 {out[label][n]['f1']:.3f} P {out[label][n]['precision']:.2f} "
                                              f"S {out[label][n]['sensitivity']:.2f} FP {out[label][n]['fp_per_case']:.2f}" for n in names), flush=True)

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
    if args.dump_decisions:
        n = write_decisions(absolute(args.dump_decisions), rows, cohort, fold, extra, kept, dumps, args.seed_offset)
        print(f"decisions: {n} lines -> {args.dump_decisions}", flush=True)


if __name__ == "__main__":
    main()
