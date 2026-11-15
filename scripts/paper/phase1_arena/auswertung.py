"""wk1 externe Baselines: verschachtelte Schwellenwahl + Auswertung.

Fuer jede Kohorte (inhouse-t2star, inhouse-swi) und das Modell SHIVA-CMB v2:
  - Wahrscheinlichkeitskarten (aus dem shiva_cmb-Lauf, _prob.nii.gz) und Referenz
    aus <private>/faelle.json (einfrorene Falten).
  - Verschachtelte Schwellenwahl: fuer jeden Fall wird NUR die Schwelle verwendet, die
    auf den vier ANDEREN Falten (Trainingsfalten) die gepoolte Touch-F1 maximiert --
    wie in der Arena (Schwelle/Mindestgroesse/Epochen kommen aus den Trainingsfalten).
    Ein Fall wird genau einmal vorhergesagt, in seiner Testfalte.
  - Bewertung bei der Modellschwelle (0.5) UND bei der verschachtelt gewaehlten Schwelle:
    cmb.metrics (Touch-Regel, 26er), gepoolt + Fall-Bootstrap 95 % CI (stratifiziert
    ueber die Kohorte, 2000 Ziehungen).
  - Subjektlevel: Zahleneinigung n_ref gegen n_pred (Spearman) und ICC(3,1)
    (einweger, ein Beoachter je Paar, mittlere Schätzung), je Kohorte und 'all'.

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
import argparse, glob, json, os
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr, pearsonr
import sys
sys.path.insert(0, _R)
from cmb.metrics import hits

C26 = np.ones((3, 3, 3), dtype=bool)
FAELLE = _CASES
GRID = np.arange(0.01, 1.0, 0.01)


def load_case(prob_path, ref_path):
    p = np.asarray(nib.load(prob_path).dataobj, dtype=np.float32)
    r = np.asarray(nib.load(ref_path).dataobj) > 0
    if r.ndim == 4:
        r = r[..., 0]
    return p, r


def f1_pool(rows):
    t = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn")}
    return 2 * t["tp"] / max(2 * t["tp"] + t["fp"] + t["fn"], 1)


def choose_threshold(train_rows, ref_all, grid=None):
    """Schwelle, die die gepoolte Touch-F1 ueber alle Trainingsfalten maximiert.
    train_rows: {case_idx: (prob, ref)}.

    Inkrementell: fuer jeden Trainingsfall werden die Komponenten des GANZEN
    Gehirns NUR EINMAL gelabelt (26er), je Komponente ihr Minimum an
    Wahrscheinlichkeit und die beruehrten Referenzkomponenten (einmal pro
    Komponente). Dann waehlt die Schwelle nur die Komponenten mit min >= t aus --
    die Touch-Zaehlung aendert sich nur, wenn t ein solches Minimum ueberspringt,
    also genuegt es, an den (wenigen) Komponenten-Minima zu testen.
    """
    from scipy.ndimage import label as ndlabel
    best_t, best_f1, best_rows = 0.5, -1.0, None
    thresholds = set()
    comp_info = {}
    for idx, (p, r) in train_rows.items():
        ref_labels, n_ref = ndlabel(r, structure=C26)
        pred_labels, n_pred = ndlabel(p > 0, structure=C26)
        info = []
        for k in range(1, n_pred + 1):
            vox = pred_labels == k
            mn = float(p[vox].min())
            touched = np.zeros(n_ref, bool)
            both = vox & r
            if both.any():
                touched[np.unique(ref_labels[both]) - 1] = True
            info.append((mn, int(touched.sum()), touched))
        comp_info[idx] = (n_ref, info)
        thresholds.update(mn for mn, _, _ in info)
    thresholds = sorted(thresholds, reverse=True)
    if not thresholds:
        return 0.5, 0.0
    for t in thresholds:
        rows = []
        for idx, (n_ref, info) in comp_info.items():
            act = [c for c in info if c[0] >= t]
            ref_hit = np.zeros(n_ref, bool)
            fp = 0
            for mn, nt, touched in act:
                if nt == 0:
                    fp += 1
                else:
                    ref_hit |= touched
            tp = int(ref_hit.sum())
            fn = n_ref - tp
            rows.append(dict(tp=tp, fp=fp, fn=fn))
        f = f1_pool(rows)
        if f > best_f1:
            best_t, best_f1, best_rows = t, f, rows
    return round(float(best_t), 4), round(float(best_f1), 4)


def bootstrap_ci(rows, n_draws=2000, seed=0):
    """Fall-Bootstrap (mit Ruecklagen, je Fall gleich gewichtet) der gepoolten F1."""
    rng = np.random.RandomState(seed)
    vals = np.array([f1_pool([rows[i] for i in rng.randint(0, len(rows), len(rows))])
                     for _ in range(n_draws)])
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def icc31(pairs):
    """ICC(3,1): ein Beoachter je Paar, mittlere Schätzung (Shrout & Fleiss 1979)."""
    x = np.array([a for a, b in pairs], float)
    y = np.array([b for a, b in pairs], float)
    n, k = len(pairs), 2
    xm, ym = x.mean(), y.mean()
    ssx = ((x - xm) ** 2).sum(); ssy = ((y - ym) ** 2).sum()
    ssxy = ((x - xm) * (y - ym)).sum()
    msb = (ssx + ssy) / (k * (n - 1))
    mse = (2 * ssxy) / (k * (n - 1))
    return float((msb - mse) / (msb + (n - 2) * mse))


def evaluate(cohort, out_dir, threshold_fixed):
    cases = json.load(open(FAELLE))[cohort]["faelle"]
    rows = []
    for c in cases:
        glob_pat = os.path.join(out_dir, f"{c['id']}_prob.nii.gz")
        p, r = load_case(glob_pat, c["maske"])
        rows.append(dict(idx=len(rows), fold=int(c["fold"]), prob=p, ref=r,
                         n_ref=int(ndimage.label(r, structure=C26)[1])))
    # verschachtelt: Schwelle je Fall aus den 4 anderen Falten
    nested_rows = []
    for row in rows:
        train = {i: (o["prob"], o["ref"]) for i, o in enumerate(rows) if o["fold"] != row["fold"]}
        t, f1_train = choose_threshold(train, None)
        nested_rows.append(dict(row, t_nested=t, f1_train=f1_train,
                                hits_nested=hits(row["ref"], row["prob"] > t)))
    fixed_rows = [dict(row, hits_nested=hits(row["ref"], row["prob"] > threshold_fixed)) for row in rows]

    def score(rw):
        pool = {k: sum(h[k] for h in (x["hits_nested"] for x in rw)) for k in ("tp", "fp", "fn")}
        n = len(rw)
        return dict(n_cases=n, tp=pool["tp"], fp=pool["fp"], fn=pool["fn"],
                    precision=round(pool["tp"] / max(pool["tp"] + pool["fp"], 1), 4),
                    sensitivity=round(pool["tp"] / max(pool["tp"] + pool["fn"], 1), 4),
                    f1=round(2 * pool["tp"] / max(2 * pool["tp"] + pool["fp"] + pool["fn"], 1), 4),
                    fp_per_case=round(pool["fp"] / n, 3),
                    f1_ci95=bootstrap_ci([x["hits_nested"] for x in rw]))
    res = dict(threshold_default=score(fixed_rows),
               threshold_nested=score(nested_rows),
               thresholds_nested={str(x["t_nested"]): x["f1_train"] for x in nested_rows})
    pairs = [(x["n_ref"], x["hits_nested"]["n_pred"]) for x in nested_rows]
    rho, _ = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
    res["subject_level"] = dict(n_ref_mean=round(float(np.mean([p[0] for p in pairs])), 2),
                                n_pred_mean=round(float(np.mean([p[1] for p in pairs])), 2),
                                spearman=round(float(rho), 4),
                                icc31=round(icc31(pairs), 4))
    # per-case rows, so a downstream 'all' pool can bootstrap properly
    res["per_case"] = {
        "default": [{"tp": x["hits_nested"]["tp"], "fp": x["hits_nested"]["fp"], "fn": x["hits_nested"]["fn"]} for x in fixed_rows],
        "nested": [{"tp": x["hits_nested"]["tp"], "fp": x["hits_nested"]["fp"], "fn": x["hits_nested"]["fn"]} for x in nested_rows],
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["t2star", "swi"], required=True)
    ap.add_argument("--dir", required=True, help="Ordner mit <id>_prob.nii.gz")
    args = ap.parse_args()
    res = evaluate(args.cohort, args.dir, 0.5)
    print(json.dumps(res, indent=1))
    out = os.path.join(os.path.dirname(args.dir), f"auswertung_shiva_{args.cohort}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("geschrieben:", out)


if __name__ == "__main__":
    main()
