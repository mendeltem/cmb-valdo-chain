"""Baut aus ~/data/sourcedata/catalina zwei saubere BIDS-Datensaetze (Symlinks, keine Kopien)
und die stratifizierte 5-fach-Falteneinteilung.

  ~/data/bids/catalina-swi      61 Faelle (1 ausgeschlossen), sub-XXX/anat/sub-XXX_acq-swi_T2starw.nii.gz
  ~/data/bids/catalina-t2star   75 Faelle (150 Ordner in der Quelle = 75 Paare byteweise
                                gleicher Zwillinge sub-XXX/sub-tXXX; genommen: sub-tXXX, die
                                in participants.tsv stehen),
                                sub-XXX/anat/sub-XXX_T2starw.nii.gz
  derivatives/manual-cmb/sub-XXX/anat/..._desc-cmb_mask.nii.gz     manuelle CMB-Maske
  derivatives/brainmask/sub-XXX/anat/..._desc-brain_mask.nii.gz    Hirnmaske (alle vorhanden)

Stratifizierung: Klassen nach CMB-Anzahl (26er-Nachbarschaft) 0 / 1-2 / 3-5 / >5, Seed 0,
je Klasse gemischt und reihum auf 5 Falten verteilt. Ergebnis in participants.tsv (Spalten
n_cmb, klasse, fold) und in ~/data/work/mb-arena/faelle.json.
Aufruf (env dl): python kohorte_bauen.py   (idempotent; vorher hirn_und_bias.py fuer fehlende Hirnmasken)"""
import os, sys, glob, json, shutil
import numpy as np, nibabel as nib, pandas as pd
from scipy import ndimage

Q = os.path.expanduser("~/data/sourcedata/catalina")
ZIEL = os.path.expanduser("~/data/bids")
ARENA = os.path.expanduser("~/data/work/mb-arena")
MOD = {  # name: (Quellordner, Unterordner, Quellsuffix, BIDS-Dateistamm-Zusatz)
    "swi": ("BIDS_SWI", "swi", "SWI", "_acq-swi_T2starw"),
    "t2star": ("BIDS_T2_STAR", "t2", "T2", "_T2starw"),
}
KLASSEN = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10**6, ">5")]
FALTEN = 5

def klasse(n):
    for lo, hi, name in KLASSEN:
        if lo <= n <= hi: return name

def versatz_mm(a, b):
    nx, ny, nz = a.shape[:3]
    c = np.array([[0,0,0,1],[nx,0,0,1],[0,ny,0,1],[0,0,nz,1],[nx,ny,0,1],[nx,0,nz,1],[0,ny,nz,1],[nx,ny,nz,1]], float).T
    return float(np.abs((a.affine@c)[:3]-(b.affine@c)[:3]).max())

def ausschluss(mod, sid, quelle, ib, ir, hirn):
    """Grund fuer Ausschluss oder None. Passt die Maske zum Bild? CMB sind dunkler als ihre
    Umgebung: mittlere Intensitaet der Maskenvoxel / mittlere Intensitaet eines Rings (Dilatation
    7x7x3 minus 3x3x1, innerhalb der Hirnmaske). Gemessen 04.09.2026: SWI P5/50/95 = 0.29/0.46/0.68,
    T2* 0.54/0.62/0.72; ein einziger SWI-Fall liegt bei 1.02 (dazu Maskenkopf 71 mm versetzt) -> raus,
    bis jemand die Maske am Bild geprueft hat. Achtung, bezahlte Falle: der Quotient zum HIRNMITTEL
    ist ohne echte Hirnmaske (Perzentil-Ersatz) unbrauchbar und hatte 11 gute Faelle verdaechtigt.
    Fehlende Hirnmasken sind kein Ausschlussgrund: hd-bet + fast -B ueber hirn_und_bias.py."""
    m = np.asarray(ir.dataobj) > 0
    if m.sum() == 0: return None
    b = np.asarray(ib.dataobj).astype(float)
    h = np.asarray(nib.load(hirn).dataobj) > 0
    ring = ndimage.binary_dilation(m, np.ones((7, 7, 3))) & ~ndimage.binary_dilation(m, np.ones((3, 3, 1))) & h
    if ring.sum() == 0: return "kein Ring in der Hirnmaske"
    q = b[m].mean() / b[ring].mean()
    if q > 0.9: return f"Maske nicht dunkler als Umgebung (Quotient {q:.2f}), passt nicht zum Bild"
    return None

def link(quelle, ziel):
    os.makedirs(os.path.dirname(ziel), exist_ok=True)
    if os.path.lexists(ziel): os.remove(ziel)
    os.symlink(os.path.abspath(quelle), ziel)

def bauen(mod):
    ordner, unter, suffix, stamm = MOD[mod]
    basis = f"{Q}/{ordner}"; ds = f"{ZIEL}/catalina-{mod}"
    # kein rmtree: derivatives/brainmask und biascorr enthalten gerechnete Dateien (hirn_und_bias.py)
    tsv = pd.read_csv(f"{basis}/participants.tsv", sep="\t")
    zeilen, notizen, raus = [], [], []
    for sid in tsv.participant_id:
        bild = f"{basis}/{sid}/{unter}/{sid}_{suffix}.nii.gz"
        roi = f"{basis}/{sid}/{unter}/{sid}_{suffix}_roi.nii.gz"
        hirn = f"{basis}/{sid}/{unter}/{sid}_{suffix}_brainmask.nii.gz"
        hirn_z = f"{ds}/derivatives/brainmask/{sid}/anat/{sid}{stamm}_desc-brain_mask.nii.gz"
        if not os.path.exists(hirn):
            # Regel des Nutzers: fehlt die Hirnmaske -> hd-bet (hirn_und_bias.py), sonst Fall zurueckstellen
            if not os.path.exists(hirn_z):
                raus.append(dict(participant_id=sid, source=tsv.set_index("participant_id").source[sid],
                                 grund="Hirnmaske fehlt noch (hirn_und_bias.py laufen lassen)")); continue
            hirn = hirn_z
        assert os.path.exists(bild) and os.path.exists(roi), sid
        ib, ir = nib.load(bild), nib.load(roi)
        assert ib.shape[:3] == ir.shape[:3], f"{sid}: Maske andere Matrix"
        quelle = tsv.set_index("participant_id").source[sid]
        grund = ausschluss(mod, sid, quelle, ib, ir, hirn)
        if grund:
            raus.append(dict(participant_id=sid, source=quelle, grund=grund)); continue
        link(bild, f"{ds}/{sid}/anat/{sid}{stamm}.nii.gz")
        m = np.asarray(ir.dataobj) > 0
        zmask = f"{ds}/derivatives/manual-cmb/{sid}/anat/{sid}{stamm}_desc-cmb_mask.nii.gz"
        v = versatz_mm(ib, ir)
        if v > 0.25 * min(ib.header.get_zooms()[:3]):
            # gleiche Matrix, nur Kopf verschoben: Maske mit dem Kopf des Bildes neu schreiben
            os.makedirs(os.path.dirname(zmask), exist_ok=True)
            nib.save(nib.Nifti1Image(m.astype(np.uint8), ib.affine, ib.header), zmask)
            notizen.append(f"{sid}: Maskenkopf {v:.1f} mm versetzt, mit Bildkopf neu geschrieben")
        else:
            link(roi, zmask)
        hat_hirn = True
        assert nib.load(hirn).shape[:3] == ib.shape[:3]
        if hirn != hirn_z: link(hirn, hirn_z)
        bias_z = f"{ds}/derivatives/biascorr/{sid}/anat/{sid}{stamm}_desc-biascorr_T2starw.nii.gz"
        _, n = ndimage.label(m, structure=np.ones((3,3,3)))
        zeilen.append(dict(participant_id=sid, source=tsv.set_index("participant_id").source[sid],
                           cmb=tsv.set_index("participant_id").cmb[sid], n_cmb=int(n), klasse=klasse(n),
                           hirnmaske="quelle" if hirn != hirn_z else "hd-bet", biaskorr="ja" if os.path.exists(bias_z) else "nein", shape="x".join(map(str, ib.shape[:3])),
                           zooms="x".join(f"{z:.3f}" for z in ib.header.get_zooms()[:3])))
    df = pd.DataFrame(zeilen)
    pd.DataFrame(raus).to_csv(f"{ds}/ausgeschlossen.tsv", sep="\t", index=False)
    rng = np.random.default_rng(0)
    df["fold"] = -1; start = 0   # reihum, Zaehler laeuft ueber die Klassen weiter -> gleich grosse Falten
    for _, _, k in KLASSEN:
        idx = df.index[df.klasse == k].to_numpy().copy(); rng.shuffle(idx)
        for i, j in enumerate(idx): df.loc[j, "fold"] = (start + i) % FALTEN
        start += len(idx)
    df.to_csv(f"{ds}/participants.tsv", sep="\t", index=False)
    json.dump({"Name": f"Catalina {mod.upper()} (CMB)", "BIDSVersion": "1.9.0", "DatasetType": "raw",
               "GeneratedBy": [{"Name": "kohorte_bauen.py", "Description": "Symlinks auf ~/data/sourcedata/catalina"}]},
              open(f"{ds}/dataset_description.json", "w"), indent=1)
    for d, name in [("manual-cmb", "manuelle CMB-Masken"), ("brainmask", "Hirnmasken")]:
        os.makedirs(f"{ds}/derivatives/{d}", exist_ok=True)
        json.dump({"Name": name, "BIDSVersion": "1.9.0", "DatasetType": "derivative",
                   "GeneratedBy": [{"Name": "kohorte_bauen.py"}]}, open(f"{ds}/derivatives/{d}/dataset_description.json", "w"), indent=1)
    with open(f"{ds}/README", "w") as f:
        f.write(f"Catalina {mod}: {len(df)} Faelle, eigene Kohorte, NICHT veroeffentlichen.\n"
                f"Bilder sind Symlinks nach {Q}/{ordner}. Masken unter derivatives/manual-cmb,\n"
                f"Hirnmasken unter derivatives/brainmask. Falten (Spalte fold) stratifiziert nach\n"
                f"CMB-Anzahl (26er-Nachbarschaft): " + ", ".join(f"{k}: {int((df.klasse==k).sum())}" for _,_,k in KLASSEN) + "\n"
                + "".join(f"Hinweis: {n}\n" for n in notizen))
    print(f"== {mod}: {len(df)} Faelle -> {ds}")
    print("   Klassen:", df.klasse.value_counts().to_dict(), " je Falte:", df.fold.value_counts().sort_index().to_dict())
    print("   Hirnmaske:", df.hirnmaske.value_counts().to_dict(), " Biaskorr:", df.biaskorr.value_counts().to_dict(), " Notizen:", len(notizen), " ausgeschlossen:", len(raus), [r["grund"][:12] for r in raus])
    with open(f"{ds}/README", "a") as f:
        f.write(f"Ausgeschlossen (ausgeschlossen.tsv): {len(raus)} Faelle.\n")
    return df, ds

faelle = {}
for mod in MOD:
    df, ds = bauen(mod)
    faelle[mod] = {"bids": ds, "n": len(df), "falten": FALTEN, "klassen": [k for _,_,k in KLASSEN],
                   "faelle": [dict(id=r.participant_id, fold=int(r.fold), n_cmb=int(r.n_cmb), klasse=r.klasse,
                                   bild=f"{ds}/{r.participant_id}/anat/{r.participant_id}{MOD[mod][3]}.nii.gz",
                                   maske=f"{ds}/derivatives/manual-cmb/{r.participant_id}/anat/{r.participant_id}{MOD[mod][3]}_desc-cmb_mask.nii.gz",
                                   hirn=f"{ds}/derivatives/brainmask/{r.participant_id}/anat/{r.participant_id}{MOD[mod][3]}_desc-brain_mask.nii.gz",
                                   bild_biaskorr=f"{ds}/derivatives/biascorr/{r.participant_id}/anat/{r.participant_id}{MOD[mod][3]}_desc-biascorr_T2starw.nii.gz" if r.biaskorr == "ja" else None)
                              for r in df.itertuples()]}
json.dump(faelle, open(f"{ARENA}/faelle.json", "w"), indent=1)
print("faelle.json geschrieben:", {m: faelle[m]["n"] for m in faelle})
