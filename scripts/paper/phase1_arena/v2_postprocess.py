"""wk1b: externe Baselines KORRIGIERT -- exaktes Post-Processing der eigenen Kette.

Auftrag wk1b-cmb-externe-baselines-korrektur (04.11.2026):
  Die alte SHIVA-Auswertung (t = 0.00, prob > 0) ist ein Metrik-Artefakt: mit der
  Touch-Regel trifft eine groesseunbegrenzte Komponente viele Referenzen -> Sensitivitaet
  0.99 / F1 0.87. Korrigerter Wert = dieselben Wahrscheinlichkeitskarten, aber mit dem
  EXAKTEN Post-Processing der eigenen Kette VOR dem Matching:
    - Komponenten (26er-Nachbarschaft) < MIN_COMPONENT_VOXELS oder > MAX_COMPONENT_VOXELS
      werden entfernt. MAX_COMPONENT_VOXELS = 2100 (cmb/stage2/candidates.py Zeile 54,
      "larger than a 10 mm sphere: not a microbleed"); minimale Groesse: dort
      voxels * 0.25 < min_mm3 mit min_mm3 = 1.0 mm3 -> MIN_COMPONENT_VOXELS = 4
      (= 2 mm3 bei 0.25 mm3/Voxel).
    - Schwellengitter 0.05..0.95 Schritt 0.05 (nie 0).
  Dazu die Komponenten-Groessenverteilung (Median, 95er-Perzentil, Anzahl) bei der
  gewaehlten Schwelle als Check.

  Je Kohorte: as-downloaded (Autoren-Schwelle 0.5) UND nested (Schwelle je Fall aus den
  4 Trainingsfalten, Gitter 0.05..0.95, maximiert gepoolte Touch-F1 mit Post-Processing),
  BEIDES mit Post-Processing. Lesionlevel P/R/F1, FP/Scan, case-bootstrap 95 % CI.
  Subjektlevel: Spearman, ICC(3,1), 4-CMB-Schwelle (Sens/Spec) nach der Logik von
  <private>/paper/patient_threshold.py (sens_spec: ref > 4, pred > 4).
  Kein neuer Inference-Run: die _prob.nii.gz-Karten des wk1-Laufs werden wiederverwendet.

  Schnelle Schwelle: jede Wahrscheinlichkeitskarte wird NUR EINMAL gelaebelt
  (prob > 0); eine Komponente wird bei Schwelle t genau dann aktiv, wenn ihr
  Komponenten-Minimum >= t und ihre Groesse in [MIN, MAX] liegt. Die Touch-Metrik
  wird dann je (Fall, t) aus den aktiven Komponenten gezaeplt -- ohne Relabeling.

Nur Summen, keine Fallkennungen in der Ausgabe.
"""
from __future__ import annotations
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import argparse, json, os, sys, time
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

sys.path.insert(0, _R)
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # sibling scripts (were in the private extern folder)
from cmb.metrics import CONNECTIVITY_26 as C26
from auswertung import bootstrap_ci, icc31

# EXAKTES Post-Processing der eigenen Kette (cmb/stage2/candidates.py):
MAX_COMPONENT_VOXELS = 2100                       # Zeile 54: groesser als Kugel 10 mm -> kein CMB
MIN_MM3, VOXEL_MM3 = 1.0, 0.25                    # Zeile 126: voxels * 0.25 < min_mm3, default 1.0
MIN_COMPONENT_VOXELS = int(np.ceil(MIN_MM3 / VOXEL_MM3))   # = 4 Voxel (= 2 mm3 bei 0.25 mm3)
GRID = [round(0.05 * i, 2) for i in range(1, 20)]          # 0.05..0.95 Schritt 0.05, NIE 0
CUT = 4                                            # 4-CMB-Schwelle (patient_threshold.py)
FAELLE = _CASES
ROOT = _P + "/extern"


def prepare_case(prob, ref):
    """Referenz-Labeling (26er) + je Gitterschwelle t: die Komponenten von prob > t DIREKT
    gelabelt (exakt, keine Naeherung), mit Groesse und beruehrten Referenzkomponenten.

    Direkt labeln statt ueber Komponenten-Minima: eine Komponente von prob > 0.05 kann bei
    hoeherer t in Unterkomponenten zerfallen (ein 0.05-Blob mit einem 0.6-Hotspot ueber
    einer CMB), und deren Groesse und Touch muss bei der jeweiligen t einzeln gezaehlt
    werden. Labeling kostet gemessen ~0.3 s je (Fall, t).
    (Nicht prob > 0 labeln: die Karten enthalten float32-Artefakte ~1.4e-45, die eine
    hirngrosse Komponente erzeugen, die jede echte Komponente verschlucken wuerde.)"""
    ref_labels, n_ref = ndimage.label(ref, structure=C26)
    per_t = {}
    for t in GRID:
        labels, n = ndimage.label(prob > t, structure=C26)
        comps = []
        for k in range(1, n + 1):
            vox = labels == k
            size = int(vox.sum())
            touched = np.zeros(n_ref, bool)
            both = vox & ref
            if both.any():
                touched[np.unique(ref_labels[both]) - 1] = True
            comps.append(dict(size=size, touched=touched))
        per_t[t] = comps
    return dict(n_ref=int(n_ref), per_t=per_t)


def case_counts(case, t):
    """Touch-Zaehlung des Falls bei Schwelle t mit Groessenfilter (exakt wie prob > t + Filter)."""
    ref_hit = np.zeros(case["n_ref"], bool)
    fp = 0
    for c in case["per_t"][t]:
        if c["size"] < MIN_COMPONENT_VOXELS or c["size"] > MAX_COMPONENT_VOXELS:
            continue
        if c["touched"].any():
            ref_hit |= c["touched"]
        else:
            fp += 1
    tp = int(ref_hit.sum())
    return dict(tp=tp, fp=int(fp), fn=case["n_ref"] - tp, n_ref=case["n_ref"],
                n_pred=int(tp + fp))


def comp_stats_at(case, t):
    """Groessenverteilung der Komponenten bei Schwelle t (VOR dem Groessenfilter
    gezaeplt wird alles; gehalten = nach Filter)."""
    kept, small, large = 0, 0, 0
    sizes = []
    for c in case["per_t"][t]:
        if c["size"] < MIN_COMPONENT_VOXELS:
            small += 1
        elif c["size"] > MAX_COMPONENT_VOXELS:
            large += 1
        else:
            kept += 1
            sizes.append(c["size"])
    return dict(n_components=kept + small + large, n_dropped_small=small, n_dropped_large=large,
                n_kept=kept, kept_voxels=int(sum(sizes)),
                kept_median_voxels=round(float(np.median(sizes)), 1) if sizes else 0.0,
                kept_p95_voxels=round(float(np.percentile(sizes, 95)), 1) if sizes else 0.0)


def score(cases, t_of, label):
    """t_of(case) -> Schwelle; gepoolte Touch-Metrik + Fall-Bootstrap 95 % CI + Subjektlevel."""
    per_case = []
    for case in cases:
        per_case.append(case_counts(case, t_of(case)))
    pool = {k: sum(x[k] for x in per_case) for k in ("tp", "fp", "fn")}
    n = len(per_case)
    ref4 = np.asarray([x["n_ref"] for x in per_case]); pred4 = np.asarray([x["n_pred"] for x in per_case])
    a, b = ref4 > CUT, pred4 > CUT
    tp4 = int((a & b).sum()); fn4 = int((a & ~b).sum())
    tn4 = int((~a & ~b).sum()); fp4 = int((~a & b).sum())
    s4 = dict(sensitivity=round(tp4 / max(tp4 + fn4, 1), 4), specificity=round(tn4 / max(tn4 + fp4, 1), 4),
              tp=tp4, fp=fp4, fn=fn4, tn=tn4, cut=CUT)
    rho, _ = spearmanr(ref4, pred4)
    return dict(label=label, n_cases=n, tp=pool["tp"], fp=pool["fp"], fn=pool["fn"],
                precision=round(pool["tp"] / max(pool["tp"] + pool["fp"], 1), 4),
                sensitivity=round(pool["tp"] / max(pool["tp"] + pool["fn"], 1), 4),
                f1=round(2 * pool["tp"] / max(2 * pool["tp"] + pool["fp"] + pool["fn"], 1), 4),
                fp_per_case=round(pool["fp"] / n, 3),
                f1_ci95=bootstrap_ci(per_case, n_draws=2000),
                subject_level=dict(n_ref_mean=round(float(ref4.mean()), 2),
                                   n_pred_mean=round(float(pred4.mean()), 2),
                                   spearman=round(float(rho), 4),
                                   icc31=round(icc31([(x["n_ref"], x["n_pred"]) for x in per_case]), 4),
                                   four_cmb_cut=s4),
                per_case=per_case)


def choose_nested(cases, counts):
    """Je Fall: Schwelle aus dem 0.05-Raster, die die gepoolte Touch-F1 der 4 Trainingsfalten
    (mit Post-Processing) maximiert. counts: {(case_idx, t): case_counts} -- vorgezaehlt."""
    idx_of = {id(c): i for i, c in enumerate(cases)}
    t_nested, f1_train = [], []
    for case in cases:
        i = idx_of[id(case)]
        best_t, best_f1 = 0.5, -1.0
        for t in GRID:
            pool = {k: sum(counts[(j, t)][k] for j in range(len(cases))
                           if cases[j]["fold"] != case["fold"]) for k in ("tp", "fp", "fn")}
            f = 2 * pool["tp"] / max(2 * pool["tp"] + pool["fp"] + pool["fn"], 1)
            if f > best_f1:
                best_t, best_f1 = t, f
        t_nested.append(best_t)
        f1_train.append(round(float(best_f1), 4))
    return t_nested, f1_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["swi", "t2star"], required=True)
    args = ap.parse_args()
    t0 = time.time()
    faelle = json.load(open(FAELLE))[args.cohort]["faelle"]
    cases = []
    for c in faelle:
        p = np.asarray(nib.load(os.path.join(ROOT, f"shiva-{args.cohort}", f"{c['id']}_prob.nii.gz")).dataobj, dtype=np.float32)
        r = np.asarray(nib.load(c["maske"]).dataobj) > 0
        if r.ndim == 4:
            r = r[..., 0]
        case = prepare_case(p, r)
        case["fold"] = int(c["fold"])
        cases.append(case)
        print(f"{c['id']}: n_ref={case['n_ref']}, n_comp={len(case['per_t'][GRID[0]])}", flush=True)
    # Alle (Fall, Schwelle)-Zahlungen NUR EINMAL vorzaehlen (136 x 19 = 2584 Billboards):
    t1 = time.time()
    counts = {}
    for i, case in enumerate(cases):
        for t in GRID:
            counts[(i, t)] = case_counts(case, t)
    print(f"precount {len(cases)}x{len(GRID)}: {round(time.time() - t1)} s", flush=True)
    t_nested, f1_train = choose_nested(cases, counts)
    for case, t in zip(cases, t_nested):
        case["t_nested"] = t
    res = dict(cohort=args.cohort, n_cases=len(cases),
               postprocessing="26-connected; keep MIN_COMPONENT_VOXELS<=voxels<=MAX_COMPONENT_VOXELS "
                              f"({MIN_COMPONENT_VOXELS}..{MAX_COMPONENT_VOXELS}); min rule = cmb/stage2/candidates.py "
                              "voxels*0.25 < min_mm3(1.0) [= 2 mm3 bei 0.25 mm3/Voxel]; MAX_COMPONENT_VOXELS=2100 "
                              "(candidates.py Zeile 54)",
               threshold_grid=GRID,
               author_threshold=score(cases, lambda c: 0.5, "as-downloaded (author threshold 0.5, postprocessed)"),
               nested=score(cases, lambda c: c["t_nested"], "nested (per-case threshold from 4 training folds, postprocessed)"),
               thresholds_nested={str(t): f for t, f in zip(t_nested, f1_train)})
    # Komponenten-Groessenverteilung bei der gewaehlten Schwelle (Check, gepoolt):
    def pool_comp(key):
        agg = {k: 0 for k in ("n_components", "n_dropped_small", "n_dropped_large", "n_kept", "kept_voxels")}
        sizes = []
        for c in cases:
            t = 0.5 if key == "a" else c["t_nested"]
            s = comp_stats_at(c, t)
            for k in agg:
                agg[k] += s[k]
            for cc in c["per_t"][t]:
                if MIN_COMPONENT_VOXELS <= cc["size"] <= MAX_COMPONENT_VOXELS:
                    sizes.append(cc["size"])
        agg["kept_median_voxels"] = round(float(np.median(sizes)), 1) if sizes else 0.0
        agg["kept_p95_voxels"] = round(float(np.percentile(sizes, 95)), 1) if sizes else 0.0
        return agg
    res["components_author"] = pool_comp("a")
    res["components_nested"] = pool_comp("n")
    res["dauer_s"] = round(time.time() - t0)
    out = os.path.join(ROOT, f"v2_shiva_{args.cohort}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("cohort", "n_cases", "dauer_s")}, indent=1))
    print("author :", json.dumps({k: res["author_threshold"][k] for k in ("f1", "precision", "sensitivity", "fp_per_case", "f1_ci95")}))
    print("nested :", json.dumps({k: res["nested"][k] for k in ("f1", "precision", "sensitivity", "fp_per_case", "f1_ci95")}))
    print("written:", out)


if __name__ == "__main__":
    main()
