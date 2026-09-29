#!/usr/bin/env python3
"""Gefaess-Uebermalung, zweite Fassung (Claude 19.09.2026, Nutzerfrage "kannst du die Uebermalung optimieren?").

Befund davor (protokoll.md 19.09.): die MicrobleedNet-Uebermalung trifft 109 von 236 Referenz-CMB, 17 sind
danach unsichtbar, und sie uebermalt 12-17 % des Hirns. Ursachen, am Quelltext gelesen
(microbleednet/scripts/data_preparation.py::inpaint_vessels):
  1. frangi(..., beta=20): beta steuert die Unterdrueckung runder Strukturen; bei 20 ist sie praktisch AUS,
     ein runder dunkler Fleck (CMB) antwortet so stark wie ein Gefaess.
  2. sigmas (0.5, 1.2, 0.2) in PIXELN: 0.45-mm- und 1.0-mm-Pixel bekommen verschiedene physische Skalen.
  3. KMeans k=2 JE SCHICHT: es gibt in jeder Schicht einen "Gefaess-Cluster", auch wenn dort kein Gefaess ist.
  4. Gerettet wird nur, was als Komponente rund UND solide ist (eccentricity < 0.9, solidity > 0.5) -- eine
     CMB, die an einem Cluster klebt, ist das nicht mehr.

Hier, ohne jede Verwendung der Label:
  a. Frangi mit beta=0.5 (runde Flecken werden unterdrueckt), Skalen in MILLIMETERN (0.5 / 1.0 / 1.5 mm).
  b. EINE Schwelle je Volumen: das oberste ANTEIL (Vorgabe 3 %) der Gefaessantwort im Hirn -- nicht je Schicht.
  c. Uebermalt wird eine Komponente (je Schicht, 8er) nur, wenn sie LANG UND DUENN ist:
     Hauptachse >= LAENGE_MM (Vorgabe 8 mm) und eccentricity >= 0.95. Alles Kurze bleibt stehen -- eine CMB
     (2-10 mm, rund) kann die Bedingung nicht erfuellen; ein quer getroffenes Gefaess (Punkt in der Schicht)
     auch nicht, das bleibt als Verwechslungskandidat bewusst im Bild.
  d. Auffuellen wie im Original (Nachbarschaftsmittel), invertieren 1 - Bild/Maximum, Hirnmaske aus dev/preproc.

  python uebermalung2.py [--faelle sub-101 ...] [--anteil 0.03] [--laenge 8] [--ziel dev/preproc_u2] [--proc 4]
  python uebermalung2.py --pruefen dev/preproc_u2     # Schadensbilanz gegen die Label (nur Summen)
"""
import os, sys, json, glob, shutil, argparse
import numpy as np, nibabel as nib
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
MBNET = "/home/uchralt/software/microbleed-detection"
N26 = np.ones((3, 3, 3), bool)


def gefaessmaske(res, hirn, px_mm, anteil, laenge_mm):
    from skimage.filters import frangi
    from skimage.measure import label, regionprops
    sig = [max(s / px_mm, 0.5) for s in (0.5, 1.0, 1.5)]
    med = float(np.median(res[hirn])) or 1.0
    v = np.zeros(res.shape, np.float32)
    for z in range(res.shape[2]):
        if not hirn[:, :, z].any():
            continue
        v[:, :, z] = frangi(res[:, :, z] / med, sigmas=sig, beta=0.5, black_ridges=True) * hirn[:, :, z]
    pos = v[hirn]; pos = pos[pos > 0]
    if len(pos) == 0:
        return np.zeros(res.shape, bool)
    thr = np.quantile(v[hirn], 1 - anteil)
    kand = (v > max(thr, 1e-12)) & hirn
    maske = np.zeros(res.shape, bool)
    for z in range(res.shape[2]):
        if not kand[:, :, z].any():
            continue
        lab = label(kand[:, :, z], connectivity=2)
        for p in regionprops(lab):
            if p.major_axis_length * px_mm >= laenge_mm and p.eccentricity >= 0.95:
                maske[:, :, z][lab == p.label] = True
    return maske


def fall(arg):
    sid, ziel, anteil, laenge = arg
    sys.path.insert(0, MBNET)
    from microbleednet.scripts.data_preparation import inpaint_with_neighborhood_mean
    d = f"{ROOT}/dev/preproc/{sid}"; z = f"{ziel}/{sid}"; os.makedirs(z, exist_ok=True)
    if os.path.exists(f"{z}/{sid}_image.nii.gz"):
        return sid, None
    ni = nib.load(f"{d}/{sid}_image.nii.gz"); hirn = np.asarray(ni.dataobj) > 0
    res = np.asarray(nib.load(f"{d}/fast_restore.nii.gz").dataobj, dtype=np.float32)
    px = float(ni.header.get_zooms()[0])
    m = gefaessmaske(res, hirn, px, anteil, laenge)
    gemalt = inpaint_with_neighborhood_mean(res.astype(np.float64), m.astype(np.float64)).astype(np.float32)
    bild = np.where(hirn, 1 - gemalt / gemalt.max(), 0).astype(np.float32)
    nib.save(nib.Nifti1Image(bild, ni.affine, ni.header), f"{z}/{sid}_image.nii.gz")
    nib.save(nib.Nifti1Image(m.astype(np.uint8), ni.affine, ni.header), f"{z}/{sid}_gefaessmaske.nii.gz")
    shutil.copy2(f"{d}/{sid}_label.nii.gz", f"{z}/{sid}_label.nii.gz")
    return sid, round(float(m.sum() / max(hirn.sum(), 1)), 4)


def pruefen(ordner):
    """Schadensbilanz wie im Protokoll 19.09.: Vergleich mit 1 - fast_restore/max (ohne Uebermalung)."""
    inv = {x["id"]: x["kohorte"] for x in json.load(open(f"{ROOT}/dev/inventar.json"))}
    ko = {k: dict(faelle=0, anteil=[], cmb=0, beruehrt=0, halb=0, ganz=0, verlust=0, unsichtbar=0) for k in "123"}
    for d in sorted(glob.glob(f"{ordner}/sub-*")):
        sid = os.path.basename(d); K = ko[inv[sid]]
        img = np.asarray(nib.load(f"{d}/{sid}_image.nii.gz").dataobj, dtype=np.float32)
        res = np.asarray(nib.load(f"{ROOT}/dev/preproc/{sid}/fast_restore.nii.gz").dataobj, dtype=np.float32)
        lab = np.asarray(nib.load(f"{d}/{sid}_label.nii.gz").dataobj) > 0
        hirn = img > 0
        erw = np.where(hirn, 1 - res / res.max(), 0)
        gemalt = hirn & (np.abs(img - erw) > 0.02)
        K["faelle"] += 1; K["anteil"].append(gemalt.sum() / max(hirn.sum(), 1))
        if not lab.any():
            continue
        ll, n = ndimage.label(lab, structure=N26)
        for c, sl in enumerate(ndimage.find_objects(ll), 1):
            s = tuple(slice(max(a.start - 4, 0), a.stop + 4) for a in sl); m = ll[s] == c
            a = gemalt[s][m].mean(); K["cmb"] += 1
            K["beruehrt"] += a > 0; K["halb"] += a >= 0.5; K["ganz"] += a == 1
            sch = ndimage.binary_dilation(m, structure=N26, iterations=3) & ~ndimage.binary_dilation(m, structure=N26) & hirn[s]
            if sch.any():
                kv = erw[s][m].mean() - erw[s][sch].mean(); kn = img[s][m].mean() - img[s][sch].mean()
                K["verlust"] += kn < 0.5 * kv; K["unsichtbar"] += kn <= 0
    ges = dict(cmb=0, beruehrt=0, halb=0, ganz=0, verlust=0, unsichtbar=0)
    for k, K in ko.items():
        if not K["faelle"]:
            continue
        print(f"Kohorte {k}: {K['faelle']} Faelle, uebermalt {100 * np.mean(K['anteil']):.1f} % des Hirns | CMB {K['cmb']}: "
              f"beruehrt {K['beruehrt']}, >= halb {K['halb']}, ganz {K['ganz']}, Kontrast > 50 % weg {K['verlust']}, unsichtbar {K['unsichtbar']}")
        for g in ges:
            ges[g] += int(K[g])
    print("gesamt:", ges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faelle", nargs="*")
    ap.add_argument("--anteil", type=float, default=0.03)
    ap.add_argument("--laenge", type=float, default=8.0)
    ap.add_argument("--ziel", default="dev/preproc_u2")
    ap.add_argument("--proc", type=int, default=4)
    ap.add_argument("--pruefen")
    a = ap.parse_args()
    if a.pruefen:
        pruefen(a.pruefen if os.path.isabs(a.pruefen) else f"{ROOT}/{a.pruefen}"); return
    ziel = a.ziel if os.path.isabs(a.ziel) else f"{ROOT}/{a.ziel}"
    ids = a.faelle or sorted(os.path.basename(d) for d in glob.glob(f"{ROOT}/dev/preproc/sub-*"))
    from multiprocessing import Pool
    with Pool(a.proc, maxtasksperchild=1) as pool:
        for sid, ant in pool.imap_unordered(fall, [(s, ziel, a.anteil, a.laenge) for s in ids]):
            print(sid, "uebersprungen" if ant is None else f"Maske {100 * ant:.2f} % des Hirns", flush=True)
    json.dump(dict(anteil=a.anteil, laenge_mm=a.laenge, sigmas_mm=[0.5, 1.0, 1.5], beta=0.5, eccentricity_min=0.95),
              open(f"{ziel}/einstellungen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
