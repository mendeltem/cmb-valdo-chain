"""Detektionsmetrik fuer Mikroblutungen: Komponenten bei 26er-Nachbarschaft.

Je Fall:  TP = Referenzkomponenten, die von >=1 Vorhersagekomponente beruehrt werden
          FN = Referenzkomponenten ohne Beruehrung
          FP = Vorhersagekomponenten, die keine Referenzkomponente beruehren
          (mehrere Vorhersagen auf einer Referenz zaehlen 1 TP und 0 FP)
Fallebene: positiv = mindestens eine Komponente -> TP/FP/TN/FN je Fall (TN gibt es nur hier).
Gesamt:   gepoolt ueber alle Faelle (Summen), F1 = 2TP/(2TP+FP+FN); dazu FP je Fall und
          mittlerer Zaehlfehler |n_vorhersage - n_referenz|.

Bibliothek:  from mb_metrik import treffer;  treffer(ref_bool, pred_bool) -> dict
Aufruf:      python mb_metrik.py --netz <ordner> --mod swi|t2star [--faelle faelle.json]
             erwartet <ordner>/fold<k>/<id>_pred.nii.gz (binaer, Bildgitter), jeder Fall genau
             einmal (in seiner Testfalte); schreibt <ordner>/ergebnis.csv (je Fall, bleibt lokal)
             und <ordner>/zusammenfassung.json (nur Summen, veroeffentlichbar).
             python mb_metrik.py --selbsttest   prueft die Logik an synthetischen Masken."""
import os, sys, glob, json, argparse
import numpy as np
from scipy import ndimage

N26 = np.ones((3, 3, 3), dtype=bool)

def treffer(ref, pred):
    ref = np.asarray(ref) > 0; pred = np.asarray(pred) > 0
    lr, nr = ndimage.label(ref, structure=N26)
    lp, npred = ndimage.label(pred, structure=N26)
    # Beruehrung = Voxelueberlappung der Komponenten
    ref_getroffen = np.zeros(nr + 1, bool); pred_getroffen = np.zeros(npred + 1, bool)
    beide = ref & pred
    if beide.any():
        ref_getroffen[np.unique(lr[beide])] = True
        pred_getroffen[np.unique(lp[beide])] = True
    tp = int(ref_getroffen[1:].sum()); fn = nr - tp; fp = int((~pred_getroffen[1:]).sum())
    return dict(n_ref=int(nr), n_pred=int(npred), tp=tp, fp=fp, fn=fn,
                fall_ref=int(nr > 0), fall_pred=int(npred > 0), zaehlfehler=abs(int(npred) - int(nr)))

def zusammenfassen(zeilen):
    s = {k: int(sum(z[k] for z in zeilen)) for k in ("n_ref", "n_pred", "tp", "fp", "fn")}
    n = len(zeilen)
    p = s["tp"] / max(s["tp"] + s["fp"], 1); r = s["tp"] / max(s["tp"] + s["fn"], 1)
    fall = dict(tp=sum(z["fall_ref"] and z["fall_pred"] for z in zeilen), fp=sum((not z["fall_ref"]) and z["fall_pred"] for z in zeilen),
                tn=sum((not z["fall_ref"]) and (not z["fall_pred"]) for z in zeilen), fn=sum(z["fall_ref"] and (not z["fall_pred"]) for z in zeilen))
    return dict(faelle=n, **s, praezision=round(p, 4), sensitivitaet=round(r, 4),
                f1=round(2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1), 4),
                fp_je_fall=round(s["fp"] / max(n, 1), 3), zaehlfehler_mittel=round(float(np.mean([z["zaehlfehler"] for z in zeilen])), 3),
                fall_ebene={k: int(v) for k, v in fall.items()},
                fall_genauigkeit=round((fall["tp"] + fall["tn"]) / max(n, 1), 4))

def selbsttest():
    def kugel(a, c, r):
        z, y, x = np.ogrid[:a.shape[0], :a.shape[1], :a.shape[2]]
        a[((z-c[0])**2 + (y-c[1])**2 + (x-c[2])**2) <= r*r] = 1
    ref = np.zeros((40, 40, 40), np.uint8); pred = np.zeros_like(ref)
    kugel(ref, (10, 10, 10), 3); kugel(ref, (25, 25, 25), 2); kugel(ref, (30, 8, 30), 2)   # 3 Referenzen
    kugel(pred, (10, 10, 10), 2); kugel(pred, (11, 13, 10), 1)   # zwei Vorhersagen auf Referenz 1 -> 1 TP, 0 FP
    kugel(pred, (25, 25, 25), 3)                                  # Referenz 2 getroffen
    pred[5, 35, 5] = 1; pred[6, 36, 6] = 1                        # nur diagonal verbunden: EIN FP bei 26er
    t = treffer(ref, pred)
    soll = dict(n_ref=3, n_pred=3, tp=2, fp=1, fn=1, fall_ref=1, fall_pred=1, zaehlfehler=0)
    assert t == soll, (t, soll)
    leer = treffer(np.zeros((5, 5, 5)), np.zeros((5, 5, 5)))
    assert leer["fall_ref"] == 0 and leer["fall_pred"] == 0 and leer["tp"] == leer["fp"] == leer["fn"] == 0
    z = zusammenfassen([t, leer, treffer(np.zeros((5, 5, 5)), pred[:5, :5, :5] * 0 + 1)])
    assert z["tp"] == 2 and z["fp"] == 2 and z["fn"] == 1 and z["fall_ebene"] == dict(tp=1, fp=1, tn=1, fn=0), z
    assert abs(z["f1"] - 4 / 7) < 1e-3, z["f1"]
    print("Selbsttest bestanden:", t, "| gesamt:", z)

def auswerten(netz, mod, faelle_pfad):
    import nibabel as nib, csv
    faelle = json.load(open(faelle_pfad))[mod]["faelle"]
    zeilen, fehlend, doppelt = [], [], []
    for f in faelle:
        preds = sorted(glob.glob(f"{netz}/fold*/{f['id']}_pred.nii.gz"))
        if not preds: fehlend.append(f["id"]); continue
        if len(preds) > 1: doppelt.append(f["id"])
        p = preds[0]; erwartet = f"{netz}/fold{f['fold']}/{f['id']}_pred.nii.gz"
        if p != erwartet: doppelt.append(f["id"] + " (falsche Falte)")
        ir, ip = nib.load(f["maske"]), nib.load(p)
        if ir.shape[:3] != ip.shape[:3]:
            raise SystemExit(f"{f['id']}: Vorhersage {ip.shape[:3]} nicht auf Bildgitter {ir.shape[:3]}")
        t = treffer(np.asarray(ir.dataobj), np.asarray(ip.dataobj))
        zeilen.append(dict(id=f["id"], fold=f["fold"], klasse=f["klasse"], **t))
    if fehlend or doppelt:
        print(f"WARNUNG: {len(fehlend)} Faelle ohne Vorhersage, {len(doppelt)} doppelt/falsche Falte:", doppelt[:5])
    if not zeilen: raise SystemExit("keine Vorhersagen gefunden")
    with open(f"{netz}/ergebnis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(zeilen[0])); w.writeheader(); w.writerows(zeilen)
    z = zusammenfassen(zeilen); z["netz"] = os.path.basename(netz.rstrip("/")); z["modalitaet"] = mod
    z["faelle_ohne_vorhersage"] = len(fehlend)
    z["je_falte"] = {k: zusammenfassen([x for x in zeilen if x["fold"] == k])["f1"] for k in sorted({x["fold"] for x in zeilen})}
    json.dump(z, open(f"{netz}/zusammenfassung.json", "w"), indent=1)
    print(json.dumps(z))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--netz"); ap.add_argument("--mod", choices=["swi", "t2star"])
    ap.add_argument("--faelle", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faelle.json"))
    ap.add_argument("--selbsttest", action="store_true"); a = ap.parse_args()
    if a.selbsttest: selbsttest()
    elif a.netz and a.mod: auswerten(a.netz, a.mod, a.faelle)
    else: ap.print_help()
