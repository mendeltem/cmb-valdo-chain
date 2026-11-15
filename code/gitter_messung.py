#!/usr/bin/env python3
"""we2 Gitter: Auswertung EINE Variante auf Falte 0 (code/gitter_messung.py).

  gitter_messung.py <nativ|gemeinsam>

Liest ergebnisse/gitter-<v>/karten/<id>_pred.nii.gz (binarisiert) und
<id>_prob.nii.gz. Die Karte liegt auf dem EINGABEGITTER der Variante:
  nativ:      Originalgitter des Falls  -> passt direkt zur Roh-CMB-Maske
  gemeinsam:  0.5 x 0.5 x 1.0 mm        -> Referenzmaske wird nearest
                                            resamplelt AUF das Gitter der Karte
Referenz = Roh-CMB-Maske (dev/cases.json). Metrik: code/metric.py
(TP/FP/FN je Fall, pooled F1/Praezision/Sensitivitaet).

Mindestgroessenwahl (ohne Testdaten): Schwelle aus den Referenzkomponenten
der TRAININGSfalten 1-4 (25. Perzentil der kleinsten CMB je Trainingsfall),
Kandidaten 0/2/4/8/16 mm3, gewaehlt = naechster Kandidat. Voxelgroesse
wird pro Fall vom GITTER DER KARTE genommen (Ko1: 0.449^2*4.0 mm3,
gemeinsam: 0.5*0.5*1.0 = 0.25 mm3) -- die Schwelle ist in mm3, also
gitterunabhaengig.

Ausgabe: ergebnisse/gitter-<v>/summary.json (Summen: gesamt + je Kohorte,
roh und gefiltert, FP je Fall als Mittel+/-Std) und per_case.csv.
Hinweis im Protokoll: n ist klein, je Kohorte ist ein Hinweis, kein Beweis.
"""
import os, sys, json, csv
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage

ROOT = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/code")
from metric import hits, summarize  # noqa: E402

KANDIDATEN_MM3 = [0, 2, 4, 8, 16]
N26 = np.ones((3, 3, 3), dtype=bool)
ZIEL_B = [0.5, 0.5, 1.0]  # (x, y, z) mm, wie gitter_bauen.py


def ko(pid):
    return "Ko1" if pid.startswith("sub-1") else ("Ko2" if pid.startswith("sub-2") else "Ko3")


def resample_ref_onto(ref, ref_path, dst_path, spacing):
    """Referenzmaske nearest resamplelen auf das Gitter der (vorhandenen)
    Kartendatei dst_path (0.5x0.5x1.0 bei 'gemeinsam')."""
    src = sitk.ReadImage(ref_path)
    dst_img = sitk.ReadImage(dst_path)
    out = sitk.Resample(src, dst_img, sitk.Transform(), sitk.sitkNearestNeighbor,
                        0, src.GetPixelID())
    # sitk.GetArrayFromImage liefert (z,y,x); nibabel-Arrays sind (x,y,z).
    # Umstellung ist ein TRANSPOSE (2,1,0), KEIN Flip -- [::-1] fuehrte zu
    # Shape-Mismatch (140,460,460) gegen (460,460,140).
    return np.transpose(np.asanyarray(sitk.GetArrayFromImage(out)), (2, 1, 0)) > 0


def filter_by_size(pred, voxmm3, mm3):
    """Komponenten unter der Groessenschwelle (mm3) entfernen."""
    if mm3 <= 0:
        return pred
    minvox = max(int(round(mm3 / voxmm3)), 1)
    if minvox <= 1:
        return pred
    lp, n = ndimage.label(pred > 0, structure=N26)
    if n == 0:
        return pred
    keep = np.array(ndimage.sum(np.ones_like(lp), lp, index=range(1, n + 1))) >= minvox
    return np.isin(lp, np.nonzero(keep)[0] + 1)


def main():
    v = sys.argv[1]
    z = f"{ROOT}/ergebnisse/gitter-{v}"
    karten = f"{z}/karten"
    cases = json.load(open(f"{ROOT}/dev/cases.json"))["t2star"]["cases"]

    # ---- Trainingsfalten 1-4: Referenzgroessen fuer die Auswahl (mm3) ----
    minima = []
    for c in cases:
        if 0 < c["fold"] <= 4:
            ref = np.asanyarray(nib.load(c["mask"]).dataobj) > 0
            voxmm3 = float(np.prod(nib.load(c["mask"]).header.get_zooms()[:3]))
            if ref.any():
                lp, n = ndimage.label(ref, structure=N26)
                sizes = np.array(ndimage.sum(ref.astype(int), lp, index=range(1, n + 1))) * voxmm3
                minima.append(float(sizes.min()))
    thr = float(np.percentile(minima, 25)) if minima else 0.0
    chosen = min(KANDIDATEN_MM3, key=lambda m: abs(m - thr))

    # ---- Falte 0: Test ----
    rows = []
    for c in cases:
        if c["fold"] != 0:
            continue
        p = f"{karten}/{c['id']}_pred.nii.gz"
        if not os.path.exists(p):
            print(f"FEHLEND: {p}", file=sys.stderr)
            continue
        img = nib.load(p)
        pred = np.asanyarray(img.dataobj) > 0
        voxmm3 = float(np.prod(img.header.get_zooms()[:3]))
        if v == "nativ":
            ref = np.asanyarray(nib.load(c["mask"]).dataobj) > 0
            assert ref.shape == pred.shape, (c["id"], ref.shape, pred.shape)
        else:
            ref = resample_ref_onto(None, c["mask"], p, ZIEL_B)
            assert ref.shape == pred.shape, (c["id"], ref.shape, pred.shape)
        pf = filter_by_size(pred, voxmm3, chosen)
        rows.append(dict(id=c["id"], ko=ko(c["id"]), voxmm3=voxmm3, n_ref=int(ref.sum()),
                         roh=hits(ref, pred), gef=hits(ref, pf)))

    def pooled(rows_, key):
        s = summarize_rows([r[key] for r in rows_])
        s["n"] = len(rows_)
        s["fp_per_case_mean_std"] = fp_ms([r[key] for r in rows_])
        return s

    def summarize_rows(rs):
        return summarize([dict(tp=r["tp"], fp=r["fp"], fn=r["fn"], n_ref=r["n_ref"],
                               n_pred=r["n_pred"], case_ref=r["case_ref"],
                               case_pred=r["case_pred"], count_error=r["count_error"])
                          for r in rs])

    def fp_ms(rs):
        fps = [r["fp"] for r in rs]
        return f"{np.mean(fps):.2f}+/-{np.std(fps):.2f}" if fps else "0.00+/-0.00"

    out = {
        "variante": v,
        "gitter": ("Originalgitter je Fall" if v == "nativ"
                   else "gemeinsam 0.5 x 0.5 x 1.0 mm (Referenz nearest resamplelt)"),
        "groessenwahl_mm3": chosen,
        "groessenwahl_basis": (f"25. Perzentil der kleinsten Referenzkomponente je Trainingsfall "
                               f"(Falten 1-4, {len(minima)} positive Faelle, "
                               f"Mittel {np.mean(minima):.2f}+/-{np.std(minima):.2f} mm3); "
                               f"Testdaten steuern die Auswahl nicht mit"),
        "gesamt": {"roh": pooled(rows, "roh"), "gefiltert": pooled(rows, "gef")},
        "je_kohorte": {k: {"roh": pooled([r for r in rows if r["ko"] == k], "roh"),
                           "gefiltert": pooled([r for r in rows if r["ko"] == k], "gef")}
                       for k in ("Ko1", "Ko2", "Ko3")},
        "hinweis": "n klein (13 Faelle): je Kohorte ist ein Hinweis, kein Beweis.",
    }
    json.dump(out, open(f"{z}/summary.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{z}/per_case.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "ko", "voxmm3", "n_ref", "n_pred_roh", "tp_roh", "fp_roh", "fn_roh",
                    "n_pred_gef", "tp_gef", "fp_gef", "fn_gef"])
        for r in rows:
            w.writerow([r["id"], r["ko"], round(r["voxmm3"], 4), r["roh"]["n_ref"],
                        r["roh"]["n_pred"], r["roh"]["tp"], r["roh"]["fp"], r["roh"]["fn"],
                        r["gef"]["n_pred"], r["gef"]["tp"], r["gef"]["fp"], r["gef"]["fn"]])
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
