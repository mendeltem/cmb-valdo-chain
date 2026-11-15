#!/usr/bin/env python3
"""Uebermalung 2 ueber alle Kohorten pruefen (Claude 19.09.2026, Nutzerfrage: muessen die Parameter fuer
inhouse-T2*, inhouse-SWI und VALDO verschieden sein?).

Je Fall wird die Gefaessantwort EINMAL gerechnet (uebermalung2: Frangi beta=0.5, Skalen 0.5/1.0/1.5 mm, je
Schicht), danach ein Gitter aus ANTEIL (oberste x % im Hirn) und LAENGE_MM (Mindestlaenge der Komponente in der
Schicht) ausgewertet -- ohne aufzufuellen, es zaehlt nur die Maske:
  maske_anteil      Anteil des Hirns, der uebermalt wuerde
  cmb_beruehrt      Referenz-CMB (26er-Komponenten) mit mindestens einem Voxel in der Maske
  cmb_halb          ... mit >= 50 % ihrer Voxel in der Maske
Dazu je Kohorte, was die Wahl der Parameter erklaert: Pixelgroesse, Schichtdicke, CMB-Durchmesser in der
Schicht (mm), Kontrast CMB gegen Umgebung.
Ausgabe NUR Summen je Kohorte: <aus>.json. inhouse ist privat -- die Datei bleibt unter <private>/.

  python uebermalung2_kohorten.py --aus <private>/uebermalung2_kohorten.json [--proc 4] [--max 0]
"""
import os, sys, json, glob, argparse
import numpy as np, nibabel as nib
from scipy import ndimage

ROOT = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/code")
N26 = np.ones((3, 3, 3), bool)
ANTEILE = (0.02, 0.03, 0.05)
LAENGEN = (6.0, 8.0, 12.0)


def faelle():
    out = []
    inv = {x["id"]: x["kohorte"] for x in json.load(open(f"{ROOT}/dev/inventar.json"))}
    val = set(json.load(open(f"{ROOT}/dev/split.json"))["validation"])
    for d in sorted(glob.glob(f"{ROOT}/dev/preproc/sub-*")):
        sid = os.path.basename(d)
        if sid in val:
            continue                                   # Validierung nicht zum Einstellen benutzen
        out.append(dict(kohorte=f"valdo-{inv[sid]}", id=sid, bild=f"{d}/fast_restore.nii.gz",
                        hirn=f"{d}/{sid}_image.nii.gz", label=f"{d}/{sid}_label.nii.gz"))
    f = json.load(open(os.environ.get("CMB_CASES", os.path.join(os.environ.get("CMB_PRIVATE", os.path.join(ROOT, "private")), "faelle.json"))))
    for mod in ("t2star", "swi"):
        for c in f[mod]["faelle"]:
            bk = c["bild"].replace(f"/{c['id']}/anat/", f"/derivatives/biascorr/{c['id']}/anat/")
            bk = glob.glob(os.path.dirname(bk) + "/*biascorr*.nii.gz")
            out.append(dict(kohorte=os.environ.get("CMB_INHOUSE_LABEL", "inhouse") + f"-{mod}", id=c["id"], bild=bk[0] if bk else c["bild"],
                            hirn=c["hirn"], label=c["maske"], biaskorr=bool(bk)))
    return out


def ein_fall(c):
    from skimage.filters import frangi
    from skimage.measure import label, regionprops
    try:
        ni = nib.load(c["bild"]); res = np.asarray(ni.dataobj, dtype=np.float32)
        if res.ndim == 4:
            res = res[..., 0]
        hirn = np.asarray(nib.load(c["hirn"]).dataobj) > 0
        if hirn.ndim == 4:
            hirn = hirn[..., 0]
        lab = np.asarray(nib.load(c["label"]).dataobj) > 0
        if lab.ndim == 4:
            lab = lab[..., 0]
        assert res.shape == hirn.shape == lab.shape, (res.shape, hirn.shape, lab.shape)
        zo = [float(z) for z in ni.header.get_zooms()[:3]]
        # Schichtachse = die mit dem groessten Voxelmass; nach hinten drehen
        ax = int(np.argmax(zo))
        if ax != 2:
            res, hirn, lab = (np.moveaxis(a, ax, 2) for a in (res, hirn, lab)); zo = [z for i, z in enumerate(zo) if i != ax] + [zo[ax]]
        px = float(np.mean(zo[:2]))
        hirn = hirn & (res > 0)
        med = float(np.median(res[hirn])) or 1.0
        sig = [max(s / px, 0.5) for s in (0.5, 1.0, 1.5)]
        v = np.zeros(res.shape, np.float32)
        for z in range(res.shape[2]):
            if hirn[:, :, z].any():
                v[:, :, z] = frangi(res[:, :, z] / med, sigmas=sig, beta=0.5, black_ridges=True) * hirn[:, :, z]
        ll, n = ndimage.label(lab, structure=N26)
        komp = ndimage.find_objects(ll)
        erg = dict(kohorte=c["kohorte"], px=px, dz=zo[2], n_cmb=int(n), hirn_ml=float(hirn.sum() * np.prod(zo) / 1000), gitter={})
        # CMB-Eigenschaften: Durchmesser in der Schicht (mm), Kontrast (dunkel) gegen Schale, relativ zum Hirnmedian
        dm, ko = [], []
        for i, sl in enumerate(komp, 1):
            s = tuple(slice(max(a.start - 4, 0), a.stop + 4) for a in sl); m = ll[s] == i
            fl = m.sum(2).max() if m.ndim == 3 else 0
            dm.append(2 * np.sqrt(m.any(2).sum() * px * px / np.pi))
            sch = ndimage.binary_dilation(m, structure=N26, iterations=3) & ~ndimage.binary_dilation(m, structure=N26) & hirn[s]
            if sch.any():
                ko.append(float((res[s][sch].mean() - res[s][m].mean()) / med))
        erg["cmb_durchmesser_mm"] = dm; erg["cmb_kontrast_rel"] = ko
        for ant in ANTEILE:
            thr = np.quantile(v[hirn], 1 - ant)
            kand = (v > max(thr, 1e-12)) & hirn
            labs = [label(kand[:, :, z], connectivity=2) if kand[:, :, z].any() else None for z in range(res.shape[2])]
            props = [regionprops(l) if l is not None else [] for l in labs]
            for lae in LAENGEN:
                maske = np.zeros(res.shape, bool)
                for z, (l, ps) in enumerate(zip(labs, props)):
                    ok = [p.label for p in ps if p.major_axis_length * px >= lae and p.eccentricity >= 0.95]
                    if ok:
                        maske[:, :, z] = np.isin(l, ok)
                ber = halb = 0
                for i, sl in enumerate(komp, 1):
                    m = ll[sl] == i; a = maske[sl][m].mean(); ber += a > 0; halb += a >= 0.5
                erg["gitter"][f"{ant}|{lae}"] = dict(maske_anteil=float(maske.sum() / max(hirn.sum(), 1)), cmb_beruehrt=int(ber), cmb_halb=int(halb))
        return erg
    except Exception as e:
        return dict(kohorte=c["kohorte"], fehler=f"{type(e).__name__}: {e}"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aus", required=True)
    ap.add_argument("--proc", type=int, default=4)
    ap.add_argument("--max", type=int, default=0, help="hoechstens so viele Faelle je Kohorte (0 = alle)")
    a = ap.parse_args()
    fl = faelle()
    if a.max:
        je = {}
        fl = [c for c in fl if je.setdefault(c["kohorte"], []).append(1) or len(je[c["kohorte"]]) <= a.max]
    from multiprocessing import Pool
    erg = []
    with Pool(a.proc, maxtasksperchild=2) as pool:
        for i, e in enumerate(pool.imap_unordered(ein_fall, fl), 1):
            erg.append(e)
            if i % 10 == 0:
                print(i, "/", len(fl), flush=True)
    aus = {}
    for ko in sorted({e["kohorte"] for e in erg}):
        E = [e for e in erg if e["kohorte"] == ko and "fehler" not in e]
        F = [e["fehler"] for e in erg if e["kohorte"] == ko and "fehler" in e]
        dm = np.array([d for e in E for d in e["cmb_durchmesser_mm"]]); kr = np.array([k for e in E for k in e["cmb_kontrast_rel"]])
        z = dict(faelle=len(E), fehler=len(F), fehlertexte=sorted(set(F))[:3], n_cmb=int(sum(e["n_cmb"] for e in E)),
                 pixel_mm=round(float(np.mean([e["px"] for e in E])), 2), schicht_mm=round(float(np.mean([e["dz"] for e in E])), 2),
                 hirn_ml=round(float(np.mean([e["hirn_ml"] for e in E]))),
                 cmb_durchmesser_mm=dict(p10=round(float(np.percentile(dm, 10)), 1), mittel=round(float(dm.mean()), 1), p90=round(float(np.percentile(dm, 90)), 1)) if len(dm) else None,
                 cmb_kontrast_rel_mittel=round(float(kr.mean()), 3) if len(kr) else None, gitter={})
        for key in (E[0]["gitter"] if E else {}):
            z["gitter"][key] = dict(maske_prozent=round(100 * float(np.mean([e["gitter"][key]["maske_anteil"] for e in E])), 2),
                                    cmb_beruehrt=int(sum(e["gitter"][key]["cmb_beruehrt"] for e in E)),
                                    cmb_halb=int(sum(e["gitter"][key]["cmb_halb"] for e in E)))
        aus[ko] = z
    json.dump(dict(anteile=ANTEILE, laengen_mm=LAENGEN, kohorten=aus), open(a.aus, "w"), indent=1, ensure_ascii=False)
    print("geschrieben:", a.aus)


if __name__ == "__main__":
    main()
