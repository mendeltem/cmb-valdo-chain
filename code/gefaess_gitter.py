#!/usr/bin/env python3
"""Gittersuche OHNE neues Training (Claude 19.09.2026, Nutzerauftrag "mit Gridsearch herausfinden"):
nuetzt Gefaess-Information ueberhaupt, und bei welchen Parametern?

Idee: statt das Bild vor dem Training zu uebermalen (je Gitterpunkt ein Training von 1.7 h), wird die
Gefaessmaske als NACHFILTER auf fertige Vorhersagen angewandt -- wie die Liquor-Regel. Eine vorhergesagte
Komponente wird verworfen, wenn mindestens UEBERLAPP ihrer Voxel in der Gefaessmaske liegen. Die Maske ist
dieselbe wie in uebermalung2 (Frangi beta 0.5, Skalen 0.5/1.0/1.5 mm, oberste ANTEIL im Hirn, Komponente in der
Schicht >= LAENGE mm und eccentricity >= 0.95), gerechnet auf dem UNUEBERMALTEN Gitterbild (dev/gitter_d, hell =
dunkel im T2*). Das ganze Gitter ANTEIL x LAENGE x UEBERLAPP kostet Minuten.
Aussage: steigt F1 nirgends, traegt die Gefaessmaske keine Information, die das Netz nicht schon hat -- dann
lohnt auch die Uebermalung nicht. Steigt es, zeigt das Gitter, WELCHE Gefaesse stoeren (lange? kurze? wie viele?).
Grenze: ein Nachfilter ist nicht dasselbe wie Uebermalen vor dem Training; die Trainingslaeufe auf
gitter_c/d/e/f/g bleiben die eigentliche Antwort, das hier ist der billige Wegweiser.
Auswahl-Hinweis: das Maximum ueber ein Gitter ist optimistisch. Berichtet wird deshalb auch die
KREUZWEISE Wahl: beste Zelle auf vier Falten, gemessen auf der fuenften.

  python gefaess_gitter.py --netz ergebnisse/turnier/a03-aniso-e60-gd [--proc 5]
  -> <netz>/gefaess_gitter.json (nur Summen)
"""
import os, sys, json, glob, argparse, itertools
import numpy as np, nibabel as nib
from scipy import ndimage

ROOT = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/code")
from metric import hits, summarize
import gewebekarte as G

ANTEILE = (0.01, 0.02, 0.03, 0.05, 0.08)
LAENGEN = (4.0, 6.0, 8.0, 12.0, 16.0)
UEBERLAPP = (0.25, 0.5)
PX = 0.5                                                  # Gitter: 0.5 x 0.5 x 1.0 mm


def ein_fall(arg):
    netz, f = arg
    from skimage.filters import frangi
    from skimage.measure import label, regionprops
    sid = os.path.basename(f)[:-len("_pred.nii.gz")]; k = int(f.split("/fold")[1][0])
    pred = np.asarray(nib.load(f).dataobj) > 0
    ref = np.asarray(nib.load(f"{ROOT}/dev/gitter_d/{sid}/{sid}_label.nii.gz").dataobj) > 0
    img = np.asarray(nib.load(f"{ROOT}/dev/gitter_d/{sid}/{sid}_image.nii.gz").dataobj, dtype=np.float32)
    seg = np.asarray(nib.load(f"{ROOT}/dev/synthseg/{sid}_synthseg.nii.gz").dataobj).astype(np.int32)
    hirn = img > 0
    # Liquor-Regel zuerst (Stand der Technik bei uns), damit der Gewinn ZUSAETZLICH dazu gemessen wird
    lp, n = ndimage.label(pred, structure=G.N26)
    obj = ndimage.find_objects(lp)
    liq = np.zeros(n + 1, bool)
    for c, sl in enumerate(obj, 1):
        w = seg[sl][lp[sl] == c]; w = w[w > 0]
        liq[c] = bool(len(w)) and int(np.bincount(w).argmax()) in G.CSF
    roh = hits(ref, pred)
    basis = hits(ref, pred & ~liq[lp])
    aus = dict(id=sid, fold=k, roh=roh, liquor=basis, gitter={})
    if n == 0 or not (~liq[1:]).any():
        for key in itertools.product(ANTEILE, LAENGEN, UEBERLAPP):
            aus["gitter"]["|".join(map(str, key))] = basis
        return aus
    # Gefaessantwort nur in den Schichten, in denen ueberhaupt eine Vorhersage liegt
    zs = sorted({z for c, sl in enumerate(obj, 1) if not liq[c] for z in range(sl[2].start, sl[2].stop)})
    v = np.zeros(img.shape, np.float32)
    sig = [s / PX for s in (0.5, 1.0, 1.5)]
    # Schwelle je Volumen braucht die Verteilung im ganzen Hirn: jede 4. Schicht reicht als Stichprobe
    probe = sorted(set(range(0, img.shape[2], 4)) | set(zs))
    for z in probe:
        if hirn[:, :, z].any():
            v[:, :, z] = frangi(img[:, :, z], sigmas=sig, beta=0.5, black_ridges=False) * hirn[:, :, z]
    vh = v[:, :, probe][hirn[:, :, probe]]
    for ant in ANTEILE:
        thr = max(float(np.quantile(vh, 1 - ant)), 1e-12)
        labs = {z: label(v[:, :, z] > thr, connectivity=2) for z in zs}
        props = {z: regionprops(l) for z, l in labs.items()}
        for lae in LAENGEN:
            maske = np.zeros(img.shape, bool)
            for z in zs:
                ok = [p.label for p in props[z] if p.major_axis_length * PX >= lae and p.eccentricity >= 0.95]
                if ok:
                    maske[:, :, z] = np.isin(labs[z], ok)
            ant_komp = np.zeros(n + 1)
            for c, sl in enumerate(obj, 1):
                m = lp[sl] == c; ant_komp[c] = maske[sl][m].mean()
            for ue in UEBERLAPP:
                weg = liq | (ant_komp >= ue); weg[0] = False
                aus["gitter"][f"{ant}|{lae}|{ue}"] = hits(ref, pred & ~weg[lp])
    return aus


def f1(rows):
    return summarize(rows)["f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netz", required=True)
    ap.add_argument("--proc", type=int, default=5)
    a = ap.parse_args()
    netz = a.netz if os.path.isabs(a.netz) else f"{ROOT}/{a.netz}"
    fl = sorted(glob.glob(f"{netz}/fold*/*_pred.nii.gz"))
    from multiprocessing import Pool
    with Pool(a.proc, maxtasksperchild=4) as pool:
        E = pool.map(ein_fall, [(netz, f) for f in fl])
    keys = list(E[0]["gitter"])
    tab = {key: summarize([e["gitter"][key] for e in E]) for key in keys}
    bestes = max(keys, key=lambda x: tab[x]["f1"])
    # kreuzweise Wahl: beste Zelle auf vier Falten, gemessen auf der fuenften
    kreuz = []
    for k in range(5):
        tr = [e for e in E if e["fold"] != k]; te = [e for e in E if e["fold"] == k]
        wahl = max(keys, key=lambda x: f1([e["gitter"][x] for e in tr]))
        kreuz += [e["gitter"][wahl] for e in te]
    s_roh, s_liq, s_kreuz = summarize([e["roh"] for e in E]), summarize([e["liquor"] for e in E]), summarize(kreuz)
    kurz = lambda s: {x: round(s[x], 4) for x in ("f1", "precision", "sensitivity", "fp_per_case")}
    aus = dict(netz=os.path.relpath(netz, ROOT), n_faelle=len(E), roh=kurz(s_roh), liquor=kurz(s_liq),
               bestes_gitter=dict(zelle=bestes, **kurz(tab[bestes])), kreuzweise_gewaehlt=kurz(s_kreuz),
               gitter={key: kurz(tab[key]) for key in keys})
    json.dump(aus, open(f"{netz}/gefaess_gitter.json", "w"), indent=1)
    print(f"{aus['netz']}: roh F1 {s_roh['f1']:.4f} | + Liquor {s_liq['f1']:.4f} | + Gefaessfilter, beste Zelle {bestes}: "
          f"{tab[bestes]['f1']:.4f} | kreuzweise gewaehlt (ehrlich): {s_kreuz['f1']:.4f}")
    for ue in UEBERLAPP:
        print(f"\nF1 bei Ueberlapp >= {ue}  (Zeilen: Anteil, Spalten: Mindestlaenge {LAENGEN} mm)")
        for ant in ANTEILE:
            print(f"  {ant:5.2f}  " + "  ".join(f"{tab[f'{ant}|{l}|{ue}']['f1']:.3f}" for l in LAENGEN))


if __name__ == "__main__":
    main()
