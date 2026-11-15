#!/usr/bin/env python3
"""Zerlegung des Deltas we3 -> cv5 (Gutachter-Punkt 3, 12.09.).

Rechnet die cv5-Wahrscheinlichkeitskarten (gecacht, <id>_pred_proba.nii.gz)
mit FIXER Schwelle 0.5 und OHNE Mindestgroessen-Filter durch -- exakt die
we3-Bewertungsregel. Damit wird der gemischte Delta zerlegt:

  A = cv5 gepoolt, fix 0.5 (Methodenanteil: Schwelle+Mindestgroesse out-of-fold)
  B = cv5 fold0, fix 0.5    (Vergleichspunkt fuer we3 = 0.0093 auf fold0)

Zahlen: code/metric.py hits() (26er), gepoolt ueber alle 5 Testfalten,
jeder Fall genau einmal. Keine neuen Vorhersagen, keine GPU.

Aufruf:  python code/delta_zerlegung.py > logs/delta_zerlegung.log 2>&1
"""
import json, sys, glob, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import hits
import nibabel as nib
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_bool(p):
    return np.asarray(nib.load(p).dataobj) > 0

def main():
    cdata = json.load(open(os.path.join(R, "dev", "cases.json")))
    caselist = cdata["t2star"]["cases"]  # {id, fold, mask, image, frst}
    ref_by_id = {c["id"]: c["mask"] for c in caselist}

    out = {}
    for konf in ["r1", "r2", "r3"]:
        base = os.path.join(R, "ergebnisse", f"cv5-{konf}")
        folds = sorted(glob.glob(os.path.join(base, "fold*")),
                       key=lambda p: int(p.split("fold")[-1]))
        rows = []
        per_fold = {}
        for fdir in folds:
            k = int(fdir.split("fold")[-1])
            probas = sorted(glob.glob(os.path.join(fdir, "*_pred_proba.nii.gz")))
            rows_fold = []
            for ppath in probas:
                cid = os.path.basename(ppath).replace("_pred_proba.nii.gz", "")
                refp = os.path.join(R, "dev", "gitter_c", cid, cid + "_label.nii.gz")
                if not os.path.exists(refp):
                    print(f"FEHLEND: kein ref {refp}", flush=True)
                    sys.exit(1)
                ref = load_bool(refp)
                # Gespeichert als uint8 = p*255 (cv5.py Zeile 348); Schwelle gilt im 0-1-Raum
                raw = np.asarray(nib.load(ppath).dataobj)
                proba01 = raw.astype(np.float32) / 255.0
                rows_fold.append((k, cid, ref, proba01))
            per_fold[k] = rows_fold
            rows.extend(rows_fold)
        # gepoolt mit fixer Schwelle 0.5, kein Filter
        res = [hits(ref, pr > 0.5) for _, _, ref, pr in rows]
        tp = sum(z["tp"] for z in res); fp = sum(z["fp"] for z in res); fn = sum(z["fn"] for z in res)
        pooled = 2 * tp / max(2 * tp + fp + fn, 1)
        # fold0 mit fix 0.5
        res0 = [hits(ref, pr > 0.5) for _, _, ref, pr in per_fold.get(0, [])]
        tp0 = sum(z["tp"] for z in res0); fp0 = sum(z["fp"] for z in res0); fn0 = sum(z["fn"] for z in res0)
        f0 = 2 * tp0 / max(2 * tp0 + fp0 + fn0, 1)
        # je Falte mit fix 0.5
        fje = []
        for k in sorted(per_fold):
            zk = [hits(ref, pr > 0.5) for _, _, ref, pr in per_fold[k]]
            t = sum(z["tp"] for z in zk); f = sum(z["fp"] for z in zk); n = sum(z["fn"] for z in zk)
            fje.append(2 * t / max(2 * t + f + n, 1))
        out[konf] = {"pooled_f1_fix05": round(pooled, 4), "fold0_f1_fix05": round(f0, 4),
                     "f1_je_falte_fix05": [round(x, 4) for x in fje],
                     "tp_fp_fn": [tp, fp, fn], "tp0_fp0_fn0": [tp0, fp0, fn0],
                     "n_faelle": len(rows)}
        print(f"{konf}: pooled(fix0.5)={pooled:.4f}  fold0(fix0.5)={f0:.4f}  "
              f"je_falte={fje}  n={len(rows)}", flush=True)
    json.dump(out, open(os.path.join(R, "dev", "delta_zerlegung.json"), "w"), indent=1)
    print("OK -> dev/delta_zerlegung.json", flush=True)

if __name__ == "__main__":
    main()
