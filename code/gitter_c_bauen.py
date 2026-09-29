#!/usr/bin/env python3
"""Gitter C (Claude 11.09.2026): ALLE 72 Faelle (57 Falten + 15 Validierung)
auf 0.5 x 0.5 x 1.0 mm, hirnbeschnitten, FRST neu (code/frst.py).

Quelle: dev/preproc/<id>/<id>_{image,label}.nii.gz (Originalgitter; Bild ist
schon invertiert 1-img/max, gefaess-uebermalt, hirnmaskiert).
Ziel:   dev/gitter_c/<id>/<id>_{image,label,frst}.nii.gz + info.json.

Unterschiede zu Gitter B (code/gitter_bauen.py):
  1. Halbvoxel-Versatz behoben. gitter_bauen.py setzte den Ursprung (= Mitte
     von Voxel 0) des Zielgitters auf den der Quelle. Bei Spacing-Wechsel
     s -> t verschiebt das das ganze Feld um (s-t)/2: in Ko1 z (4.0 -> 1.0 mm)
     um 1.5 mm, und die letzten Zielschichten liegen hinter der letzten
     Quellschicht (Index 34.75 > 34.5 = Rand -> SimpleITK-Default 0: leere
     Schichten). Hier bleibt die physische MITTE des Sichtfelds fest:
       Quellindex i(j) = off + r*j,  r = t/s,  off = ((N-1) - r*(M-1)) / 2
     damit liegen alle Zielvoxel-Mitten innerhalb [-0.5, N-0.5] der Quelle
     (Ko1 z: i in [-0.375, 34.375]); am Rand wird die Randschicht gehalten
     (mode nearest), nicht 0 eingesetzt.
  2. Zuschnitt auf die Hirn-Bounding-Box (Bild != 0) + 8 Voxel Rand (auf
     das Volumen gekappt); die Affine wird um den Versatz verschoben, die
     Weltkoordinaten bleiben exakt.
  3. FRST wird auf dem Gitter-C-Bild NEU gerechnet (nicht das alte FRST
     interpoliert), Radien [2,3,4,6] vx = 1-3 mm in allen Kohorten.
Resampling rein im Indexraum (scipy.ndimage.affine_transform, Diagonale +
Versatz), Affine = A_quelle @ T: keine LPS/RAS-Umrechnung, schiefe Affinen
(Ko1, Ko3) bleiben exakt erhalten. Achsreihenfolge wie die Quelle (x,y,z);
der Lader transponiert.

Pruefung je Fall: 26er-Komponentenzahl Label Quelle == Ziel (voll) == Ziel
(beschnitten), alle Laesionsvoxel im Zuschnitt, Affinen image/label/frst
identisch. Die FRST-Guete (Mittel auf CMB vs Hirn, Anteil Komponenten mit
max FRST im oberen 1 % des Hirns) wird NUR fuer die 57 Falten-Faelle des
gueltigen Splits (dev/split.json, alter Seed-0-Stand) gerechnet; die 15
Validierungsfaelle werden gebaut, aber nicht ausgewertet (nur Form/Zuschnitt/
Komponentenzahl-Gleichheit als Gitterpruefung). Der erste Lauf am 11.09.
lief waehrend des (verworfenen) Neusplits und hat die Guete fuer 13 Faelle
ausgelassen; sammeln() traegt sie aus den gespeicherten Dateien nach.

Wiederaufnehmbar: ein Fall mit info.json gilt als fertig. Max. 8 Prozesse,
je Prozess 1 Thread, ein Fall je Prozess (maxtasksperchild=1).

  python gitter_c_bauen.py [--faelle sub-101 ...] [--proc 8] [--nur-sammeln]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only (GPU-Pause)
import sys, json, time, argparse, resource
import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frst as frstmod

ROOT = "/home/uchralt/data/work/valdo-t2s"
QUELLE = os.environ.get("GITTER_C_QUELLE", f"{ROOT}/dev/preproc")   # 19.09.: Variante OHNE Gefaess-Uebermalung (dev/preproc_roh)
ZIEL = os.environ.get("GITTER_C_ZIEL", f"{ROOT}/dev/gitter_c")   # Umlenkung nur fuer Probelaeufe
ORDNUNG = int(os.environ.get("GITTER_C_ORDNUNG", "1"))     # Interpolationsordnung des BILDES (Label bleibt naechster Nachbar)
SPACING = np.array([float(v) for v in os.environ.get("GITTER_C_SPACING", "0.5,0.5,1.0").split(",")])   # 19.09.: z = 2 mm als Variante pruefbar
RAND = 8                                   # Voxel (Zielgitter)
NEU = os.environ.get("GITTER_C_NEU") == "1"   # --neu setzt das fuer die Arbeitsprozesse
S26 = np.ones((3, 3, 3), dtype=bool)


def n_komp(lab):
    return int(ndimage.label(lab > 0, structure=S26)[1])


def label_nearest(lab, r, off, M):
    """Label nearest auf das Zielgitter, mit Komponenten-Buchfuehrung.
    Nearest beim Verkleinern (Ko1 in der Ebene 0.449 -> 0.5 mm, Ko2 z 0.8 ->
    1.0 mm) laesst jede ~9./5. Quellzeile aus: eine 1 Voxel duenne Komponente
    kann dabei ganz verschwinden (gemessen 11.09.: 1 von 236, eine 3-Voxel-
    Linie in einer einzigen Zeile). Solche Komponenten werden gerettet: ihr
    Indikator wird im eigenen Kasten LINEAR auf das Zielgitter gebracht und
    die Zielvoxel >= 0.5 x Maximum gesetzt (Fussabdruck wie bei nearest,
    in Ko1 z also 4 Schichten, statt eines Einzelvoxels). Zusaetzlich gezaehlt:
    Zielkomponenten mit >1 Quellkomponente (Verschmelzung) und Quell-
    komponenten, die in >1 Zielkomponente zerfallen."""
    lk, n = ndimage.label(lab > 0, structure=S26)
    tk = ndimage.affine_transform(lk.astype(np.int32), r, offset=off, output_shape=tuple(M),
                                  order=0, mode="nearest", output=np.int32)
    fehlend = sorted(set(range(1, n + 1)) - set(np.unique(tk).tolist()))
    gerettet = []
    if fehlend:
        kaesten = ndimage.find_objects(lk)
        for k in fehlend:
            ks = kaesten[k - 1]
            i0 = np.array([max(q.start - 2, 0) for q in ks])
            i1 = np.array([min(q.stop + 2, lk.shape[d]) for d, q in enumerate(ks)])
            ind = (lk[tuple(slice(a, b) for a, b in zip(i0, i1))] == k).astype(np.float32)
            j0 = np.maximum(np.floor((i0 - off) / r).astype(int), 0)
            j1 = np.minimum(np.ceil((i1 - 1 - off) / r).astype(int) + 1, np.array(M))
            w = ndimage.affine_transform(ind, r, offset=off + r * j0 - i0, output_shape=tuple(j1 - j0),
                                         order=1, mode="constant", cval=0.0, output=np.float32)
            sub = tk[tuple(slice(a, b) for a, b in zip(j0, j1))]
            setz = (w >= 0.5 * w.max()) & (sub == 0)
            sub[setz] = k
            gerettet.append({"quellvoxel": int(ind.sum()), "zielvoxel": int(setz.sum()),
                             "zielkasten": [int(v) for v in j0]})
    del lk
    lvoll = (tk > 0).astype(np.uint8)
    tl, tn = ndimage.label(lvoll, structure=S26)
    paare = np.unique(np.stack([tl[lvoll > 0], tk[lvoll > 0]]), axis=1)   # (Zielkomp, Quellkomp)
    verschm = int((np.bincount(paare[0]) > 1).sum())
    zerfall = int((np.bincount(paare[1]) > 1).sum())
    return lvoll, {"verloren_gerettet": gerettet, "verschmolzen": verschm, "zerfallen": zerfall}


def zielgitter(aff, shape):
    """Zielgroesse, Schritt r (Quellvoxel je Zielvoxel) und Versatz off so,
    dass die Mitte des Sichtfelds fest bleibt."""
    s = np.sqrt((aff[:3, :3] ** 2).sum(0))          # echte Voxelgroesse aus der Affine
    N = np.array(shape[:3], dtype=float)
    M = np.rint(N * s / SPACING).astype(int)
    r = SPACING / s
    off = ((N - 1) - r * (M - 1)) / 2.0
    return s, M, r, off


def speichern(arr, aff, ref_hdr, pfad, dtype):
    hdr = ref_hdr.copy()
    hdr.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(arr, dtype=dtype), aff, hdr)
    img.set_sform(aff, code=int(ref_hdr["sform_code"]) or 1)
    img.set_qform(aff, code=int(ref_hdr["qform_code"]))
    img.header.set_zooms(tuple(float(z) for z in SPACING))
    nib.save(img, pfad)


def frst_guete(fr, hirn, lab):
    """Mittel FRST auf CMB-Voxeln vs Hirn; Anteil 26er-Komponenten, deren
    max FRST im oberen 1 % der Hirnvoxel liegt."""
    lk, n = ndimage.label(lab > 0, structure=S26)
    if n == 0:
        return {"frst_hirn_mittel": round(float(fr[hirn].mean()), 5)}
    q99 = float(np.percentile(fr[hirn], 99))
    mx = ndimage.maximum(fr, lk, index=np.arange(1, n + 1))
    return {"frst_hirn_mittel": round(float(fr[hirn].mean()), 5),
            "frst_cmb_mittel": round(float(fr[lab > 0].mean()), 5),
            "frst_q99_hirn": round(q99, 5),
            "komp_top1": int((np.asarray(mx) >= q99).sum()), "komp_n": int(n)}


def fall(args):
    pid, fold, auswerten = args
    t0 = time.time()
    dst = f"{ZIEL}/{pid}"
    if os.path.exists(f"{dst}/info.json") and not NEU:
        return json.load(open(f"{dst}/info.json"))
    os.makedirs(dst, exist_ok=True)
    try:
        qi = nib.load(f"{QUELLE}/{pid}/{pid}_image.nii.gz")
        ql = nib.load(f"{QUELLE}/{pid}/{pid}_label.nii.gz")
        if not np.allclose(qi.affine, ql.affine, atol=1e-5) or qi.shape != ql.shape:
            raise RuntimeError("Quelle: image/label nicht auf einem Gitter")
        A = qi.affine
        s, M, r, off = zielgitter(A, qi.shape)
        img = qi.get_fdata(dtype=np.float32)
        lab = (np.asarray(ql.dataobj) > 0).astype(np.uint8)
        n_quelle = n_komp(lab)
        vox_quelle = int(lab.sum())
        t_laden = time.time() - t0

        # --- volles Zielgitter (Mitte fest), dann Hirn-Bounding-Box
        t1 = time.time()
        voll = ndimage.affine_transform(img, r, offset=off, output_shape=tuple(M),
                                        order=1, mode="nearest", output=np.float32)
        if ORDNUNG != 1:
            # 19.09.: linear verschmiert kleine CMB (Kontrast -12 bis -32 % gemessen). Hoehere Ordnung haelt
            # den Kontrast besser, schwingt aber ueber: auf [0, 1] kappen, und die Hirnmaske bleibt die der
            # linearen Fassung (sonst franst der Rand aus und "Bild != 0" stimmt nicht mehr).
            hoch = ndimage.affine_transform(img, r, offset=off, output_shape=tuple(M),
                                            order=ORDNUNG, mode="nearest", output=np.float32)
            voll = np.where(voll > 0, np.clip(hoch, 1e-6, 1.0), 0).astype(np.float32)
            del hoch
        lvoll, labinfo = label_nearest(lab, r, off, M)
        del img
        nz = np.nonzero(voll)
        lo = np.maximum(np.array([a.min() for a in nz]) - RAND, 0)
        hi = np.minimum(np.array([a.max() for a in nz]) + RAND + 1, M)
        del nz
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        bild = np.ascontiguousarray(voll[sl]); del voll
        label = np.ascontiguousarray(lvoll[sl])
        n_voll = n_komp(lvoll); vox_voll = int(lvoll.sum()); del lvoll
        n_crop = n_komp(label); vox_crop = int(label.sum())
        # Affine: Zielindex j -> Quellindex off + r*j -> Welt; Zuschnitt verschiebt j um lo
        T = np.eye(4); T[:3, :3] = np.diag(r); T[:3, 3] = off + r * lo
        Ac = A @ T
        t_resample = time.time() - t1

        # --- FRST auf dem Gitter-C-Bild
        t2 = time.time()
        hirn = frstmod.hirnmaske(bild)
        fr, finfo = frstmod.frst_volumen(bild, hirn)
        t_frst = time.time() - t2

        speichern(bild, Ac, qi.header, f"{dst}/{pid}_image.nii.gz", np.float32)
        speichern(label, Ac, qi.header, f"{dst}/{pid}_label.nii.gz", np.uint8)
        speichern(fr, Ac, qi.header, f"{dst}/{pid}_frst.nii.gz", np.float32)
        # Kontrolle aus der Datei: Affinen konsistent, Label {0,1}
        aff = [nib.load(f"{dst}/{pid}_{k}.nii.gz").affine for k in ("image", "label", "frst")]
        aff_ok = all(np.allclose(a, aff[0], atol=1e-6) for a in aff) and np.allclose(aff[0], Ac, atol=1e-4)
        # Weltkontrolle: Mitte des Sichtfelds (voll) = Quelle
        mitte_q = (A @ np.r_[(np.array(qi.shape[:3]) - 1) / 2.0, 1])[:3]
        T0 = np.eye(4); T0[:3, :3] = np.diag(r); T0[:3, 3] = off
        mitte_z = (A @ T0 @ np.r_[(M - 1) / 2.0, 1])[:3]

        info = {
            "id": pid, "kohorte": pid[4], "fold": fold,
            "quelle_shape": list(qi.shape[:3]), "quelle_spacing": [round(float(v), 5) for v in s],
            "voll_shape": [int(v) for v in M], "shape": list(bild.shape),
            "crop_lo_zielvoxel": [int(v) for v in lo], "crop_hi_zielvoxel": [int(v) for v in hi],
            "crop_offset_quellvoxel": [round(float(v), 4) for v in off + r * lo],
            "affine": np.round(Ac, 6).tolist(),
            "mitte_abweichung_mm": round(float(np.abs(mitte_q - mitte_z).max()), 6),
            "affinen_konsistent": bool(aff_ok),
            "label_werte": sorted(int(v) for v in np.unique(label)),
            "n_cmb_quelle": n_quelle, "n_cmb_voll": n_voll, "n_cmb": n_crop,
            "komponenten_gleich": bool(n_quelle == n_voll == n_crop),
            "label_nearest": labinfo,
            "laesionsvoxel_quelle": vox_quelle, "laesionsvoxel_voll": vox_voll,
            "laesionsvoxel_crop": vox_crop, "alle_laesionen_im_crop": bool(vox_voll == vox_crop),
            "voxel_voll": int(np.prod(M)), "voxel_crop": int(np.prod(bild.shape)),
            "hirnvoxel": int(hirn.sum()),
            "bild_hirn_mittel": round(float(bild[hirn].mean()), 4),
            "frst": {"hirn_median_bild": finfo["hirn_median_bild"], "p999_roh": finfo["p999_roh"],
                     "radien": finfo["radien"], "modus": finfo["modus"]},
            "zeit_s": {"laden": round(t_laden, 1), "resample_crop": round(t_resample, 1),
                       "frst": round(t_frst, 1), "gesamt": round(time.time() - t0, 1)},
            "ram_max_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
        }
        info["frst_guete_gerechnet"] = bool(auswerten)
        if auswerten:                               # Validierung (alt ODER neu): nicht anschauen
            info["frst"].update(frst_guete(fr, hirn, label))
        json.dump(info, open(f"{dst}/info.json", "w"), indent=1)
        print(f"{pid} ko{pid[4]} fold {fold:>2}  {info['quelle_shape']} -> voll {info['voll_shape']} "
              f"-> crop {info['shape']} ({info['voxel_crop']/info['voxel_voll']:.2f})  "
              f"CMB {n_quelle}/{n_voll}/{n_crop}  zeit {info['zeit_s']}  RAM {info['ram_max_mb']:.0f} MB",
              flush=True)
        return info
    except Exception as e:
        import traceback
        json.dump({"id": pid, "fehler": repr(e), "trace": traceback.format_exc()},
                  open(f"{dst}/fehler.json", "w"), indent=1)
        print(f"{pid}: FEHLER {e!r}", flush=True)
        return {"id": pid, "fehler": repr(e)}


def nachtragen(pid, fold):
    """Falte aus dem gueltigen Split setzen; fehlende FRST-Guete fuer
    Falten-Faelle aus den gespeicherten Dateien nachrechnen. Validierung:
    Guete wird nie gerechnet (und falls vorhanden, nicht angefasst)."""
    p = f"{ZIEL}/{pid}/info.json"
    info = json.load(open(p))
    if "fehler" in info:
        return info
    neu = False
    if info.get("fold") != fold:
        if "fold_split_neu" in info:              # erster Lauf: Falte des verworfenen Neusplits
            info["fold_verworfener_neusplit"] = info.pop("fold_split_neu")
        info["fold"] = fold; neu = True
    if fold >= 0 and not info.get("frst_guete_gerechnet"):
        d = f"{ZIEL}/{pid}"
        b = nib.load(f"{d}/{pid}_image.nii.gz").get_fdata(dtype=np.float32)
        f = nib.load(f"{d}/{pid}_frst.nii.gz").get_fdata(dtype=np.float32)
        l = np.asarray(nib.load(f"{d}/{pid}_label.nii.gz").dataobj)
        info["frst"].update(frst_guete(f, frstmod.hirnmaske(b), l))
        info["frst_guete_gerechnet"] = True; info["frst_guete_nachgetragen"] = True
        neu = True
    if fold < 0:
        info["frst_guete_gerechnet"] = False
    if neu:
        json.dump(info, open(p, "w"), indent=1)
    return info


def sammeln(ids, fold_of):
    faelle = {}
    for pid in ids:
        p = f"{ZIEL}/{pid}/info.json"
        if os.path.exists(p):
            faelle[pid] = nachtragen(pid, fold_of[pid])
    ok = [f for f in faelle.values() if "fehler" not in f]
    zus = {
        "erzeugt": time.strftime("%Y-%m-%d %H:%M"), "skript": "code/gitter_c_bauen.py",
        "ziel_spacing_mm": SPACING.tolist(), "rand_voxel": RAND,
        "quelle": "dev/preproc/<id>/<id>_{image,label}.nii.gz",
        "frst": {"modul": "code/frst.py", "radien_voxel": list(frstmod.RADIEN), "alpha": frstmod.ALPHA,
                 "k_n": "9.9 (n=1) sonst 8", "sigma": "0.25 n", "modus": "hell (Bild invertiert)",
                 "normierung": "je Fall 99.9-Perzentil im Hirn, auf [0,1] gekappt"},
        "faelle_fertig": len(ok), "faelle_erwartet": len(ids),
        "komponenten_abweichend": [f["id"] for f in ok if not f["komponenten_gleich"]],
        "komponenten_gerettet": {f["id"]: f["label_nearest"]["verloren_gerettet"] for f in ok
                                 if f.get("label_nearest", {}).get("verloren_gerettet")},
        "komponenten_verschmolzen_oder_zerfallen": [f["id"] for f in ok if f.get("label_nearest") and
                                                    (f["label_nearest"]["verschmolzen"] or f["label_nearest"]["zerfallen"])],
        "laesion_ausserhalb_crop": [f["id"] for f in ok if not f["alle_laesionen_im_crop"]],
        "affinen_inkonsistent": [f["id"] for f in ok if not f["affinen_konsistent"]],
        "voxel_voll_summe": int(sum(f["voxel_voll"] for f in ok)),
        "voxel_crop_summe": int(sum(f["voxel_crop"] for f in ok)),
        "faelle": faelle,
    }
    alt = f"{ROOT}/dev/gitter_c_info.json"
    if os.path.exists(alt):                       # Vergleich (--vergleich) nicht verlieren
        v = json.load(open(alt)).get("frst_vergleich_alt_b")
        if v:
            zus["frst_vergleich_alt_b"] = v
    json.dump(zus, open(f"{ROOT}/dev/gitter_c_info.json", "w"), indent=1, ensure_ascii=False)
    return zus


VERGLEICH = ["sub-101", "sub-103", "sub-216", "sub-228", "sub-305", "sub-316"]   # je Kohorte 2,
# positiv, in BEIDEN Splits in einer Falte (keine Validierung, alt oder neu)


def vergleich(ids):
    """FRST-Guete neu (gitter_c, eigene FRST) gegen alt (gitter_b: altes FRST
    vom Originalgitter linear interpoliert). Hirn = Bild > 0 (Loecher je
    Schicht gefuellt) auf dem jeweiligen Gitter."""
    erg = {}
    for pid in ids:
        e = {}
        for name, d in (("alt_b", f"{ROOT}/dev/gitter_b/{pid}"), ("neu_c", f"{ZIEL}/{pid}")):
            b = nib.load(f"{d}/{pid}_image.nii.gz").get_fdata(dtype=np.float32)
            f = nib.load(f"{d}/{pid}_frst.nii.gz").get_fdata(dtype=np.float32)
            l = np.asarray(nib.load(f"{d}/{pid}_label.nii.gz").dataobj) > 0
            h = frstmod.hirnmaske(b)
            g = frst_guete(f, h, l)
            g["verhaeltnis_cmb_hirn"] = round(g["frst_cmb_mittel"] / g["frst_hirn_mittel"], 2)
            e[name] = g
            del b, f, l, h
        erg[pid] = e
        print(pid, "  ".join(f"{k}: CMB/Hirn {v['verhaeltnis_cmb_hirn']:.2f} "
                              f"top1% {v['komp_top1']}/{v['komp_n']}" for k, v in e.items()), flush=True)
    summe = {k: {"komp_top1": sum(e[k]["komp_top1"] for e in erg.values()),
                 "komp_n": sum(e[k]["komp_n"] for e in erg.values()),
                 "verhaeltnis_mittel": round(float(np.mean([e[k]["verhaeltnis_cmb_hirn"]
                                                            for e in erg.values()])), 2)}
             for k in ("alt_b", "neu_c")}
    print("Summe:", summe)
    return {"faelle": erg, "summe": summe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faelle", nargs="*")
    ap.add_argument("--proc", type=int, default=8)
    ap.add_argument("--nur-sammeln", action="store_true")
    ap.add_argument("--neu", action="store_true", help="fertige Faelle neu bauen (Dateien werden ueberschrieben)")
    ap.add_argument("--vergleich", nargs="*", help="FRST alt (gitter_b) vs neu; ohne Liste: die 6 Standardfaelle")
    a = ap.parse_args()
    cases = json.load(open(f"{ROOT}/dev/cases.json"))["t2star"]["cases"]
    fold_of = {c["id"]: c["fold"] for c in cases}
    alle = sorted(fold_of)
    unberuehrt = set(json.load(open(f"{ROOT}/dev/split.json"))["validation"])
    assert len(unberuehrt) == 15 and all(fold_of[p] == -1 for p in unberuehrt)
    assert len(alle) == 72
    if a.vergleich is not None:
        vids = a.vergleich or VERGLEICH
        assert not set(vids) & unberuehrt, "Vergleich nie auf Validierungsfaellen"
        v = vergleich(vids)
        p = f"{ROOT}/dev/gitter_c_info.json"
        z = json.load(open(p)) if os.path.exists(p) else {}
        v["ids"] = vids
        z.setdefault("frst_vergleich_alt_b", {})[f"liste{len(vids)}" if a.vergleich else "standard6"] = v
        json.dump(z, open(p, "w"), indent=1, ensure_ascii=False)
        return
    ids = a.faelle or alle
    os.makedirs(ZIEL, exist_ok=True)
    if a.neu:
        os.environ["GITTER_C_NEU"] = "1"
        global NEU
        NEU = True
    if not a.nur_sammeln:
        t0 = time.time()
        # grosse Faelle (Ko2) zuerst, damit am Ende keine Nachzuegler laufen
        jobs = sorted(((p, fold_of[p], p not in unberuehrt) for p in ids),
                      key=lambda x: (x[0][4] != "2", x[0]))
        print(f"{len(jobs)} Faelle, {min(a.proc, 8)} Prozesse", flush=True)
        from multiprocessing import Pool
        with Pool(min(a.proc, 8), maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(fall, jobs):
                pass
        print(f"gesamt {time.time() - t0:.0f} s", flush=True)
    if ZIEL != f"{ROOT}/dev/gitter_c":
        print("Probelauf (GITTER_C_ZIEL gesetzt): dev/gitter_c_info.json wird nicht geschrieben")
        return
    z = sammeln(alle, fold_of)
    print(f"fertig {z['faelle_fertig']}/72; Komponenten abweichend: {z['komponenten_abweichend']}; "
          f"Laesion ausserhalb: {z['laesion_ausserhalb_crop']}; Affinen: {z['affinen_inkonsistent']}; "
          f"Voxel voll {z['voxel_voll_summe']/1e6:.0f} M -> crop {z['voxel_crop_summe']/1e6:.0f} M")


if __name__ == "__main__":
    main()
