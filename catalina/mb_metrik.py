"""Detection metric for microbleeds: components with 26-neighbourhood connectivity.

Per case:  TP = reference components touched by >=1 prediction component
           FN = reference components without a touch
           FP = prediction components that touch no reference component
           (several predictions on one reference count as 1 TP and 0 FP)
Case level: positive = at least one component -> TP/FP/TN/FN per case (TN only exists here).
Total:    pooled over all cases (sums), F1 = 2TP/(2TP+FP+FN); plus FP per case and
          mean count error |n_pred - n_ref|.

Library:  from mb_metrik import treffer;  treffer(ref_bool, pred_bool) -> dict
Call:     python mb_metrik.py --netz <folder> --mod swi|t2star [--faelle faelle.json]
          expects <folder>/fold<k>/<id>_pred.nii.gz (binary, image grid), each case exactly
          once (in its test fold); writes <folder>/ergebnis.csv (per case, stays local)
          and <folder>/zusammenfassung.json (sums only, publishable).
          python mb_metrik.py --selbsttest   checks the logic on synthetic masks."""
import os, sys, glob, json, argparse
import numpy as np
from scipy import ndimage

N26 = np.ones((3, 3, 3), dtype=bool)

def treffer(ref, pred):
    ref = np.asarray(ref) > 0; pred = np.asarray(pred) > 0
    lr, nr = ndimage.label(ref, structure=N26)
    lp, npred = ndimage.label(pred, structure=N26)
    # touch = voxel overlap of the components
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
    kugel(ref, (10, 10, 10), 3); kugel(ref, (25, 25, 25), 2); kugel(ref, (30, 8, 30), 2)   # 3 references
    kugel(pred, (10, 10, 10), 2); kugel(pred, (11, 13, 10), 1)   # two predictions on reference 1 -> 1 TP, 0 FP
    kugel(pred, (25, 25, 25), 3)                                  # reference 2 touched
    pred[5, 35, 5] = 1; pred[6, 36, 6] = 1                        # only diagonally connected: ONE FP at 26-connectivity
    t = treffer(ref, pred)
    soll = dict(n_ref=3, n_pred=3, tp=2, fp=1, fn=1, fall_ref=1, fall_pred=1, zaehlfehler=0)
    assert t == soll, (t, soll)
    leer = treffer(np.zeros((5, 5, 5)), np.zeros((5, 5, 5)))
    assert leer["fall_ref"] == 0 and leer["fall_pred"] == 0 and leer["tp"] == leer["fp"] == leer["fn"] == 0
    z = zusammenfassen([t, leer, treffer(np.zeros((5, 5, 5)), pred[:5, :5, :5] * 0 + 1)])
    assert z["tp"] == 2 and z["fp"] == 2 and z["fn"] == 1 and z["fall_ebene"] == dict(tp=1, fp=1, tn=1, fn=0), z
    assert abs(z["f1"] - 4 / 7) < 1e-3, z["f1"]
    print("Self-test passed:", t, "| total:", z)

def auswerten(netz, mod, faelle_pfad):
    import nibabel as nib, csv
    faelle = json.load(open(faelle_pfad))[mod]["faelle"]
    zeilen, fehlend, doppelt = [], [], []
    for f in faelle:
        preds = sorted(glob.glob(f"{netz}/fold*/{f['id']}_pred.nii.gz"))
        if not preds: fehlend.append(f["id"]); continue
        if len(preds) > 1: doppelt.append(f["id"])
        p = preds[0]; erwartet = f"{netz}/fold{f['fold']}/{f['id']}_pred.nii.gz"
        if p != erwartet: doppelt.append(f["id"] + " (wrong fold)")
        ir, ip = nib.load(f["maske"]), nib.load(p)
        if ir.shape[:3] != ip.shape[:3]:
            raise SystemExit(f"{f['id']}: prediction {ip.shape[:3]} not on image grid {ir.shape[:3]}")
        t = treffer(np.asarray(ir.dataobj), np.asarray(ip.dataobj))
        zeilen.append(dict(id=f["id"], fold=f["fold"], klasse=f["klasse"], **t))
    if fehlend or doppelt:
        print(f"WARNING: {len(fehlend)} cases without a prediction, {len(doppelt)} duplicate/wrong fold:", doppelt[:5])
    if not zeilen: raise SystemExit("no predictions found")
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
