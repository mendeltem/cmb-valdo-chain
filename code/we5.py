#!/usr/bin/env python3
"""we5: Nachbearbeitung (26er-Filter in mm3 + FP-Filter ueber T1-SynthSeg) und EINMALIGE
Validierungsmessung auf dem Validierungsset von we4 (VALDO Task 2, T2*).

Harte Regeln (Auftrag + Abnahme 11.09.):
  - Alle Filterparameter werden VORHER auf den CV-Falten (57 Faellen) festgelegt und
    eingefroren in dev/filterkette.json. Das Validierungsset (15 Faelle) wird danach
    genau EINMAL beruehrt. Wer danach noch am Filter dreht, hat das Set verbrannt.
  - Groessenfilter in mm3 (Voxel x pixdim, je Fall) -- ein Voxel-Schwellenwert waere
    kohortenabhaengig (Kohorten haben sehr verschiedene Voxelvolumina).
  - FP-Filter ueber T1: mri_synthseg auf sub-XXX_space-T2S_desc-masked_T1.nii.gz.
    KEINE Registrierung -- der T1 liegt im T2S-Raum (72/72 geprueft, Vertrag we1).
    ABER: der T1 liegt im ORIGINAL-T2S-Grid, NICHT im Gitter-C-Raum der Vorhersagen
    (0.5x0.5x1.0 mm, zugeschnitten). Der SynthSeg-Label wird daher mit naechster
    Nachbarschaft (kein Interpolation, Label bleibt diskret) auf das Gitter-C-Raster
    umgerechnet -- eine Rasterumrechnung, keine Registrierung.
    Zwei Varianten, getrennt gemessen, jede mit TP-Verlust:
      V1 Ventrikel:  Labels 4,12,13,22,23 (lat./infer. lat. Ventrikel li/re, 3., 4.)
      V2 +CSF:       V1 + Label 17 (freier CSF)
    (Labeltabelle aus mri_synthseg --vol-CSV, 33 Strukturen, geprueft an sub-201;
    wd6-Vergleichswert: SWI-Kohorte, dort CSF-Variante kostete 12 TP -- hier neu messen.)
  - Stufen nur behalten, wenn sie auf der CV (gepoolt, 57 Faelle) F1 verbessern.
  - Validierung: EINMAL, bestes Modell aus we4 = Konf r3, 5 Falten-Modelle, je mit der
    out-of-fold-Schwelle und out-of-fold-Voxel-Mindestgroesse; Mehrheitsvotum (>=3 von 5).
  - Referenz = gitter_c-Label (26er-Komponenten), exakt wie we4 gemessen (label_for).
  - Zahlen als Mittel+-Std, nie Median. Gemessen/interpretiert getrennt.

Aufruf:
  python we5.py --probe             # 2 Faelle: 1 CV + 1 Val durch ALLE Stufen, Zeit messen
  python we5.py --synthseg [ids]    # mri_synthseg (CPU, parallel) + Resample auf Gitter-C
  python we5.py --cv                # Filterabstimmung auf 57 CV-Vorhersagen ->
                                    # dev/filterkette.json (EINFRIEREN)
  python we5.py --validierung       # EINMALIG: 15 Val-Faelle -> ergebnisse/validierung/
                                    # summary.json (mit filterkette_sha256)

Dateien:
  dev/synthseg/<id>_synthseg.nii.gz           SynthSeg im Gitter-C-Raum (resampled)
  dev/filterkette.json                        eingefrorene Kette (vor der Validierung!)
  ergebnisse/validierung/<id>_pred.nii.gz     Vorhersage je Val-Fall (genau eine)
  ergebnisse/validierung/summary.json         Validierungszahlen + CV daneben
"""
import os, sys, json, time, hashlib, argparse, subprocess, glob, shutil
import numpy as np
import nibabel as nib
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
EXTERN = "/home/uchralt/data/extern/valdo2021/Task2"
N26 = np.ones((3, 3, 3), dtype=bool)
FS = "/home/uchralt/freesurfer/8.2.0/bin/mri_synthseg"
V1_VENTRIKEL = {4, 12, 13, 22, 23}          # lat./infer. lat. V. (li/re), 3., 4. Ventrikel
V2_CSF = V1_VENTRIKEL | {17}                 # + freier CSF
MAX_KOMP_VOX = 2100                            # Klumpen-Waechter aus we4 (in Voxel)
MM3_GRID = (1, 2, 3, 5, 8, 12, 20, 30, 50)    # Groessenfilter in mm3 (je Fall: Voxel x pixdim)
MM3_TOP = (2, 5, 8)                            # nur diese 3 Punkte kriegen das Aspekt-Grid
ASPEKT_GRID = (1.25, 1.5, 2.0, 3.0)            # Formfilter: max/min bbox-Halblaenge

sys.path.insert(0, f"{ROOT}/code")
from metric import hits, summarize
from metric_valdo import label_for

# ------------------------------------------------------------------- Filter-Grundoperationen

def komponenten(bin_map):
    """26er-Komponenten als Liste von (voxelzahl, bbox_halblangen)."""
    lab, n = ndimage.label(bin_map, structure=N26)
    out = []
    for c in range(1, n + 1):
        m = lab == c
        z = np.argwhere(m)
        out.append((int(m.sum()), (z.max(0) - z.min(0) + 1) / 2.0))
    return lab, out

def aspekt(bb_halb):
    bb = np.maximum(bb_halb, 1.0)
    return float(bb.max() / bb.min())

def filter_mind_mm3(bin_map, vox_mm3, mind_mm3, max_komp_vox=MAX_KOMP_VOX, aspekt_max=None):
    """26er-Komponenten entfernen: < mind_mm3, > max_komp_vox Voxel, oder aspekt > aspekt_max."""
    lab, comps = komponenten(bin_map)
    out = np.zeros_like(bin_map)
    for c in range(1, lab.max() + 1):
        vox, bb = comps[c - 1]
        if vox > max_komp_vox:
            continue
        if vox * vox_mm3 < mind_mm3:
            continue
        if aspekt_max is not None and aspekt(bb) > aspekt_max:
            continue
        out |= (lab == c)
    return out

def fp_filter_synthseg(pred_bin, vent_mask):
    """Kandidatenkomponenten verwerfen, deren SCHWERPUNKT in der Ventrikel/CSF-Maske faellt.
    (Auftrag: 'deren Schwerpunkt in Ventrikel/CSF faellt'; kein Touch-Filter.)"""
    lab, n = ndimage.label(pred_bin, structure=N26)
    out = np.zeros_like(pred_bin)
    for c in range(1, n + 1):
        cen = ndimage.center_of_mass(lab == c)
        if vent_mask[tuple(np.round(cen).astype(int))]:
            continue
        out |= (lab == c)
    return out

def resample_labels(lab, src_aff, gc_aff, gc_shape):
    """Labels im Quellraster (src_aff) auf das Gitter-C-Raster (naechste Nachbarschaft,
    cval=0 = Hintergrund). Rasterumrechnung via Weltkoordinaten, keine Registrierung."""
    grids = np.meshgrid(*[np.arange(s, dtype=float) for s in gc_shape], indexing="ij")
    xyz = np.stack([g.ravel() for g in grids], 0)
    world = gc_aff @ np.vstack([xyz, np.ones(len(xyz[0]))[None, :]])
    src_vox = np.linalg.inv(src_aff) @ world
    # src_vox ist (4, N) in homogenen Koordinaten; nur die ersten 3 Achsen resamplen.
    return ndimage.map_coordinates(lab, src_vox[:3].reshape(3, *gc_shape),
                                   order=0, mode="constant", cval=0).astype(np.uint8)

def synthseg_maske(pid, variant):
    """Resample SynthSeg (T1-Grid) auf Gitter-C (naechste Nachbarschaft), Boolean-Maske."""
    seg = nib.load(f"{ROOT}/dev/synthseg/{pid}_synthseg.nii.gz")
    gc = nib.load(f"{ROOT}/dev/gitter_c/{pid}/{pid}_image.nii.gz")
    lab = np.asarray(seg.dataobj)
    if lab.shape != gc.shape:
        raise SystemExit(f"{pid}: synthseg {lab.shape} != gitter_c {gc.shape}")
    return np.isin(lab, list(variant))

def kette_anwenden(pred_bin, vox_mm3, kette, vent_mask=None):
    """Eingefrorene Kette auf eine binaere Vorhersage anwenden (Gitter-C-Raum)."""
    out = pred_bin
    if kette.get("groesse_mm3"):
        out = filter_mind_mm3(out, vox_mm3, kette["groesse_mm3"],
                              aspekt_max=kette.get("aspekt"))
    if kette.get("synthseg_variant") and vent_mask is not None:
        out = fp_filter_synthseg(out, vent_mask)
    return out

# ------------------------------------------------------------------- SynthSeg-Aufruf

def synthseg_lauf(ids, parallel=3):
    """mri_synthseg (CPU, 8 Threads pro Fall) + Resample auf Gitter-C."""
    os.makedirs(f"{ROOT}/dev/synthseg", exist_ok=True)
    zu_tun = [i for i in ids if not os.path.exists(f"{ROOT}/dev/synthseg/{i}_synthseg.nii.gz")]
    if not zu_tun:
        print("alle SynthSeg schon da"); return
    def ein(id_):
        t1 = f"{EXTERN}/{id_}/{id_}_space-T2S_desc-masked_T1.nii.gz"
        o = f"{ROOT}/dev/synthseg/{id_}_roh"
        t0 = time.time()
        # NaN-Reparatur (Coach 14.09. 21:5x): 33/72 gelieferte T1s tragen NaN als
        # Randschale ausserhalb des Hirnraums (dreifach erodierte Hirnmaske NaN-frei,
        # sub-304: eine zusammenhaengende Komponente an der Volumenrand, sub-102: zwei).
        # NaN vergiftet die Perzentilnormierung des Segmentierers -> leere
        # Segmentierung, rc 0, Fall verschwindet lautlos. Nullsetzen erfindet keine
        # Anatomie. Bereinigte TEMPORAERE Kopie; die gelieferte Datei bleibt unberuehrt.
        im = nib.load(t1)
        a = np.asanyarray(im.dataobj)
        nan0 = t1 + ".nan0.nii.gz"
        if np.isnan(a).any():
            nib.save(nib.Nifti1Image(np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0),
                                     im.affine), nan0)
            t1 = nan0
        r = subprocess.run([FS, "--i", t1, "--o", o, "--cpu", "--threads", "8", "--noaddctab"],
                           capture_output=True, text=True)
        if os.path.exists(nan0):
            os.remove(nan0)
        if r.returncode != 0:
            return id_, f"rc={r.returncode}: {r.stderr[:200]}"
        f = glob.glob(f"{o}/*_synthseg.nii.gz")
        if not f:
            return id_, "keine Ausgabe"
        # neueste Datei (mtime), nicht alphabetisch: eine stale Rohausgabe aus einem
        # fruheren Lauf (z.B. vor der NaN-Reparatur) darf die aktuelle nicht verdecken
        f = [max(f, key=os.path.getmtime)]
        roh = nib.load(f[0])
        gc = nib.load(f"{ROOT}/dev/gitter_c/{id_}/{id_}_image.nii.gz")
        zi = resample_labels(np.asarray(roh.dataobj), roh.affine, gc.affine, gc.shape)
        u = np.unique(zi)
        if len(u) < 20:
            return id_, f"SUSP: nur {len(u)} Labels nach Resample"
        nib.save(nib.Nifti1Image(zi, gc.affine), f"{ROOT}/dev/synthseg/{id_}_synthseg.nii.gz")
        # Rohausgabe nach Erfolg entfernen (rmtree: mri_synthseg hinterlaesst mehr
        # als eine Datei im Verzeichnis, os.rmdir scheitert mit ENOTEMPTY)
        shutil.rmtree(o, ignore_errors=True)
        return id_, f"ok {time.time()-t0:.0f}s labels={len(u)}"
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for id_, st in ex.map(ein, zu_tun):
            print(f"synthseg {id_}: {st}", flush=True)

# ------------------------------------------------------------------- CV: Vorhersagen + Messung

def cv_faelle():
    """57 CV-Faelle: (id, bin_pred, vox_mm3, ref) auf Gitter-C."""
    d = json.load(open(f"{ROOT}/dev/split.json"))
    out = []
    for k, fids in enumerate(d["folds"]):
        for pid in fids:
            pp = f"{ROOT}/ergebnisse/cv5-best/fold{k}/{pid}_pred.nii.gz"
            if not os.path.exists(pp):
                raise SystemExit(f"CV-Vorhersage fehlt: {pp}")
            pi = nib.load(pp)
            ref = np.asarray(label_for(pid, pi).dataobj) > 0
            out.append((pid, np.asarray(pi.dataobj) > 0,
                        float(abs(np.linalg.det(pi.affine[:3, :3]))), ref))
    assert len(out) == 57, len(out)
    return out

def messen(faelle, kette=None, ventmasken=None):
    """Gepoolte 26er-Messung; ventmasken: {pid: {variant: boolarray}} oder None."""
    rows = []
    for pid, pm, vox_mm3, ref in faelle:
        out = pm
        if kette:
            vm = ventmasken[pid][kette["synthseg_variant"]] if kette.get("synthseg_variant") else None
            out = kette_anwenden(pm, vox_mm3, kette, vent_mask=vm)
        rows.append(dict(id=pid, **hits(ref, out)))
    return summarize(rows), rows

def cv_abstimmung(faelle, ventmasken):
    """Alle Stufen auf der CV messen, beste Kette (nur F1-verbessernde Stufen)."""
    roh, _ = messen(faelle)
    print(f"CV roh: F1 {roh['f1']} sens {roh['sensitivity']} fp/fall {roh['fp_per_case']} "
          f"(n_pred {roh['n_pred']})", flush=True)

    beste = (roh["f1"], dict(groesse_mm3=None, aspekt=None, synthseg_variant=None))
    grid_z = {}
    for mm3 in MM3_GRID:
        asps = ASPEKT_GRID if mm3 in MM3_TOP else [None]
        for asp in asps:
            s, _ = messen(faelle, dict(groesse_mm3=mm3, aspekt=asp, synthseg_variant=None))
            grid_z[f"{mm3}mm3_a{asp}"] = s["f1"]
            if s["f1"] > beste[0]:
                beste = (s["f1"], dict(groesse_mm3=mm3, aspekt=asp, synthseg_variant=None))
    b_f1, b_k = beste
    print(f"CV beste Groessenstufe: F1 {b_f1} kette {b_k}", flush=True)

    # SynthSeg-Stufe: auf der besten Groessenstufe, V1 und V2 getrennt
    stufen = {"roh": roh}
    stufen["groesse"] = messen(faelle, dict(groesse_mm3=b_k["groesse_mm3"],
                                            aspekt=b_k["aspekt"], synthseg_variant=None))[0] \
        if b_k["groesse_mm3"] else roh
    for v in ("V1", "V2"):
        k = dict(b_k, synthseg_variant=v)
        s, _ = messen(faelle, k, ventmasken)
        stufen[v] = s
        print(f"CV {v}: F1 {s['f1']} sens {s['sensitivity']} fp/fall {s['fp_per_case']} "
              f"tp {s['tp']} (Basis tp {stufen['groesse']['tp']})", flush=True)
        if s["f1"] > beste[0]:
            beste = (s["f1"], k)

    kette = beste[1]
    # TP-Verluste je SynthSeg-Variante ausweisen (gegen Groessenstufe)
    tp_verlust = {v: stufen["groesse"]["tp"] - stufen[v]["tp"] for v in ("V1", "V2")}
    print(f"EINGEFROREN: F1 {beste[0]} kette {kette} TP-Verlust {tp_verlust}", flush=True)

    return dict(
        kette=kette,
        cv_roh=roh,
        cv_groesse=stufen["groesse"],
        cv_v1=stufen["V1"],
        cv_v2=stufen["V2"],
        tp_verlust_v1=tp_verlust["V1"],
        tp_verlust_v2=tp_verlust["V2"],
        groesse_grid_f1=grid_z,
        entscheidung=("Stufe behalten nur wenn sie auf der CV (gepoolt, 57 Faelle) F1 "
                      "verbessert; Groessenfilter in mm3 je Fall (Voxel x pixdim); "
                      "SynthSeg-Filter: Schwerpunkt der Kandidatenkomponente in "
                      "Ventrikel(V1)/Ventrikel+CSF(V2); V1 und V2 getrennt gemessen."),
        eingefroren_am=time.strftime("%Y-%m-%d %H:%M:%S"),
        hinweis=("Kette auf CV-Falten abgestimmt und EINGEFROREN, bevor das "
                 "Validierungsset (15 Faelle) angefasst wurde. Weiteres Nachjustieren "
                 "nach dem Blick auf die Validierungszahlen ist verboten (Set verbrannt)."),
    )

# ------------------------------------------------------------------- Validierung (EINMAL)

def validierung_einmal(fk_pf, probe=False, schwelle=None):
    import torch
    import cv5
    cv5.DEV = os.environ.get("WE5_DEV", "cuda")
    fk = json.load(open(fk_pf))
    kette = fk["kette"]
    val_ids = json.load(open(f"{ROOT}/dev/split.json"))["validation"]
    if probe:
        val_ids = val_ids[:2]
        outd = f"{ROOT}/dev/we5_probe"
    else:
        outd = f"{ROOT}/ergebnisse/validierung"
    os.makedirs(outd, exist_ok=True)

    # 5 r3-Falten-Modelle (bestes Modell aus we4) mit out-of-fold-Parametern
    mods = []
    for k in range(5):
        meta = json.load(open(f"{ROOT}/dev/modelle/best/fold{k}/meta.json"))
        assert meta["konf"] == "r3"
        m, nch = cv5.neues_modell(meta["params"])
        m.load_state_dict(torch.load(f"{ROOT}/dev/modelle/best/fold{k}/modell.pt", map_location="cpu"))
        m.to(cv5.DEV); m.eval()
        mods.append((m, meta["schwelle"], meta["mindestgroesse"]))
    print(f"{len(mods)} r3-Modelle geladen (Schwellen "
          f"{[m[1] for m in mods]}, mg-Voxel {[m[2] for m in mods]})", flush=True)

    cv5.lade_daten(ids=None)   # nur Falten-Faelle; Validierung wird nicht geladen (fold -1)
    # Val-Faelle in die recs-Struktur bringen (gleiche Gitter-C-Dateien)
    d = json.load(open(f"{ROOT}/dev/cases.json"))["t2star"]["cases"]
    for c in d:
        if c["fold"] != -1:
            continue
        sid = c["id"]
        ni = nib.load(f"{cv5.GITTER}/{sid}/{sid}_image.nii.gz")
        dreh = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))
        cv5.FA["recs"][sid] = dict(
            img=dreh(np.asarray(ni.dataobj, dtype=np.float32)),
            frs=dreh(np.asarray(nib.load(f"{cv5.GITTER}/{sid}/{sid}_frst.nii.gz").dataobj, dtype=np.float32)),
            lab=dreh((np.asarray(nib.load(f"{cv5.GITTER}/{sid}/{sid}_label.nii.gz").dataobj) > 0).astype(np.uint8)),
            komp=[], fold=-1, affine=ni.affine)

    # SynthSeg-Masken (V1/V2) im Gitter-C-Raum
    ventmasken = {pid: {"V1": synthseg_maske(pid, V1_VENTRIKEL), "V2": synthseg_maske(pid, V2_CSF)}
                  for pid in val_ids}
    # Praez-Check der Resample-Maske
    for pid in val_ids[:1]:
        print(f"{pid} synthseg-Maske: V1 {int(ventmasken[pid]['V1'].sum())} Voxel, "
              f"V2 {int(ventmasken[pid]['V2'].sum())} Voxel", flush=True)

    rows = []
    t0 = time.time()
    for i, pid in enumerate(val_ids):
        gc = nib.load(f"{ROOT}/dev/gitter_c/{pid}/{pid}_image.nii.gz")
        vox_mm3 = float(abs(np.linalg.det(gc.affine[:3, :3])))
        votes = np.zeros_like(np.asarray(gc.dataobj), dtype=np.uint8)
        for m, thr, mg_vox in mods:
            p, nk = cv5.proba_fall(m, pid)
            # proba_fall liefert cv5-Internausrichtung (x,y,z) (recs via dreh=transpose(2,1,0));
            # votes/ventmasken liegen im nativen Gitter-C-Raum (z,y,x) -> zurueckdrehen wie cv5.py Z.340/347
            p = np.ascontiguousarray(p.transpose(2, 1, 0))
            b = p > (schwelle if schwelle is not None else thr)
            # out-of-fold Voxel-Mindestgroesse (we4: in Voxel)
            lab, n = ndimage.label(b, structure=N26)
            b2 = np.zeros_like(b)
            for c in range(1, n + 1):
                if mg_vox <= (lab == c).sum() <= MAX_KOMP_VOX:
                    b2 |= (lab == c)
            votes += b2
        pred_bin = votes >= 3
        pred_fin = kette_anwenden(pred_bin, vox_mm3, kette, vent_mask=ventmasken[pid][kette["synthseg_variant"]] if kette.get("synthseg_variant") else None)
        nib.save(nib.Nifti1Image(pred_fin.astype(np.uint8), gc.affine), f"{outd}/{pid}_pred.nii.gz")
        ref = np.asarray(label_for(pid, nib.load(f"{outd}/{pid}_pred.nii.gz")).dataobj) > 0
        t = hits(ref, pred_fin)
        rows.append(dict(id=pid, **t))
        print(f"[{i+1}/{len(val_ids)}] {pid}: ref={t['n_ref']} pred={t['n_pred']} "
              f"tp={t['tp']} fp={t['fp']} fn={t['fn']}", flush=True)
        torch.cuda.empty_cache()

    z = summarize(rows)
    z["cases"] = len(val_ids)
    z["n_pred"] = int(sum(r["n_pred"] for r in rows))
    z["filterkette_sha256"] = hashlib.sha256(open(fk_pf, "rb").read()).hexdigest()
    z["modell"] = ("we4-Sieger r3: 5 Falten-Modelle, je out-of-fold-Voxel-Mindestgroesse, "
                   "Mehrheitsvotum >=3/5"
                   + (f", Schwelle {schwelle}" if schwelle is not None
                      else ", je out-of-fold-Schwelle"))
    z["filterkette"] = kette
    z["cv_daneben"] = dict(roh=fk["cv_roh"], groesse=fk["cv_groesse"],
                           v1=fk["cv_v1"], v2=fk["cv_v2"])
    z["unsicherheit"] = ("47 Referenz-Blutungen tragen etwa +-0.1 F1 Unsicherheit; "
                         "starke Abweichung zur CV ist ein Befund, keine Aufforderung "
                         "zur Nachjustierung (das Set wurde genau einmal beruehrt).")
    z["dauer_s"] = round(time.time() - t0, 1)
    z["probe"] = probe
    tmp = f"{outd}/summary.json.tmp"
    json.dump(z, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, f"{outd}/summary.json")
    print("VALIDIERUNG FERTIG ->", f"{outd}/summary.json", flush=True)
    print(json.dumps(z, indent=1, ensure_ascii=False), flush=True)
    return z

# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--validierung", action="store_true")
    ap.add_argument("--synthseg", nargs="*", default=None)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--schwelle", type=float, default=None,
                    help="gilt fuer --validierung: globale Schwelle statt je-Falte out-of-fold-Schwelle")
    a = ap.parse_args()
    if a.synthseg is not None:
        synthseg_lauf(a.synthseg, a.parallel)
    elif a.cv:
        faelle = cv_faelle()
        print(f"{len(faelle)} CV-Faelle geladen", flush=True)
        ventmasken = {pid: {"V1": synthseg_maske(pid, V1_VENTRIKEL),
                            "V2": synthseg_maske(pid, V2_CSF)} for pid, *_ in faelle}
        fk = cv_abstimmung(faelle, ventmasken)
        tmp = f"{ROOT}/dev/filterkette.json.tmp"
        json.dump(fk, open(tmp, "w"), indent=1, ensure_ascii=False)
        os.replace(tmp, f"{ROOT}/dev/filterkette.json")
        print("dev/filterkette.json EINGEFROREN", flush=True)
    elif a.probe:
        # 1 CV-Fall (Filter-Pfade) + 2 Val-Faelle (Inferenz-Zeit), mit Zeitmessung
        faelle = cv_faelle()[:1]
        pid = faelle[0][0]
        v = {"V1": synthseg_maske(pid, V1_VENTRIKEL), "V2": synthseg_maske(pid, V2_CSF)}
        t0 = time.time()
        roh, _ = messen(faelle)
        t1 = time.time()
        s, _ = messen(faelle, dict(groesse_mm3=5, aspekt=None, synthseg_variant=None), {"V1": v, "V2": v})
        s, _ = messen(faelle, dict(groesse_mm3=5, aspekt=None, synthseg_variant="V2"), {"V1": v, "V2": v})
        t2 = time.time()
        print(f"PROBE CV-Fall {pid}: roh F1 {roh['f1']}, Filter {t1-t0:.2f}s, +SynthSeg {t2-t1:.2f}s")
        validierung_einmal(f"{ROOT}/dev/filterkette.json", probe=True)
        print("PROBE OK", flush=True)
    elif a.validierung:
        validierung_einmal(f"{ROOT}/dev/filterkette.json", schwelle=a.schwelle)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
