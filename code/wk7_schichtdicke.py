#!/usr/bin/env python3
"""wk7 (07.11.2026): CMB-Detektion bei simulierter Schichtdicke -- Auswertung.

Auftrag Step 4: die VALDO-5-Fold-Modelle sagen ihre EIGENE Testfalte (out-of-fold,
keine Leakage) bei jeder Dicke aus: native gitter_d + gitter_thick_3/5/7. Feste
Nachbearbeitung wie cmb.transfer.evaluate (Schwelle 0.3, mind 2 mm3, max 2100 Voxel,
26er) -- die Schwelle wird NICHT je Dicke neu gewaehlt. Zusaetzlich wird die je-Falte
innerfold-Schwelle (meta.json "schwelle") als zweiter Betriebspunkt ausgewertet.

Modelle: base a03-aniso-e60-gd (the workstation, Seed 42), s42 und s43 aus
ergebnisse/turnier_hpc_wk7 (HPC, 06.11., 10/10 COMPLETED). Beide HPC-Seeds werden
BERICHTET, nie mit the workstation-Falten gemischt.

Metriken je Dicke (27 RSS-Faelle, Kohorte 2, eingefrorene Falten):
  Läsions-F1/Sens/Prec, FP/Scan (pooled, 26er, wie code/metric.py),
  Sensitivitaet nach Referenzgroesse <2 / 2-3 / 3-5 / >5 mm (maximaler Durchmesser
  der Komponente auf dem jeweiligen Gitter), Patientenlevel:
  (a) Pearson + Spearman Rank-Korrelation der Fall-Zaehlungen (Ref vs Pred),
  (b) Sens/Spec bei >4 CMB (positiv = >=5 Referenz-Komponenten).
Gepaarter Bootstrap ueber Faelle (1000 Iterationen, Resampling mit Ruckgabe):
  Delta (thick - native) je Metrik mit 95 % CI, je Modell und Dicke.

CPU (CV5_DEV=cpu; Nutzerregel 04.11.2026: kein lokales GPU fuer CMB).
Vorhergesagt wird JE Fall in seiner Testfalte mit dem Modell dieser Falte.
Wiederaufnehmbar: vorhandene _pred_proba.nii.gz werden nicht neu berechnet.

    CV5_DEV=cpu python code/wk7_schichtdicke.py --modelle NAME=PFAD ... --out ergebnisse/schichtdicke
"""
import os, sys, json, argparse, time
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy import stats as scistats

ROOT = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/code")
from metric import hits, summarize        # 26er, wie cmb.transfer.evaluate

THRESHOLD, MIN_MM3, MAX_VOXELS, VOXEL_MM3 = 0.3, 2.0, 2100, 0.25
N26 = np.ones((3, 3, 3), bool)
NEN = ("native", "n3", "n5", "n7")
MAN = {
    "native": f"{ROOT}/dev/manifest_valdo_gitter_d.json",
    "n3": f"{ROOT}/dev/manifest_valdo_gitter_thick_3.json",
    "n5": f"{ROOT}/dev/manifest_valdo_gitter_thick_5.json",
    "n7": f"{ROOT}/dev/manifest_valdo_gitter_thick_7.json",
}


def faelle(dicke, ko="valdo-2"):
    """Faelle eines Gitters, auf die Kohorte gefiltert.

    WICHTIG: das native Manifest (gitter_d) deckt drei Kohorten ab (57 Faelle:
    valdo-1/2/3), die Dicken-Manifeste nur valdo-2 (27). Ohne diesen Filter laedt
    auswerten() Referenzen aus fremden Kohorten. Jeder Eintrag traegt sein eigenes
    "dir" (gitter_d bzw. gitter_thick_N) -- nicht den Ordner des Manifestpfads raten.
    """
    return [e for e in json.load(open(MAN[dicke])) if e.get("cohort") == ko]


def post_process(prob, thr, min_mm3=MIN_MM3, max_vox=MAX_VOXELS, vox_mm3=VOXEL_MM3):
    m = prob > thr
    if not m.any():
        return m
    lab, n = ndimage.label(m, structure=N26)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    keep = (size * vox_mm3 >= min_mm3) & (size <= max_vox)
    keep[0] = False
    return keep[lab]


def komponenten_groesse_mm(lab_bool, aff):
    """maximaler Durchmesser (mm) je 26er-Komponente."""
    lab, n = ndimage.label(lab_bool, structure=N26)
    if n == 0:
        return []
    s = np.sqrt((aff[:3, :3] ** 2).sum(0))
    out = []
    for k in range(1, n + 1):
        c = np.argwhere(lab == k)
        d = (c.max(0) - c.min(0) + 1) * s
        out.append(float(d.max()))
    return out


def groessen_klasse(mm):
    return "<2" if mm < 2 else "2-3" if mm < 3 else "3-5" if mm <= 5 else ">5"


def vorhersagen(models, net, nen):
    """<out>/<modell>/<dicke>/fold<k>/<id>_pred_proba.nii.gz (uint8, p*255).
    Ein Fall genau in seiner Testfalte; vorhandene Karten werden behalten."""
    import cv5
    import turnier
    from netze import NETZE
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    out = {}
    for name, mpfad in models:
        nets = []
        for k in range(5):
            nw = NETZE[net](2)
            nw.load_state_dict(torch_load(f"{mpfad}/fold{k}/modell.pt"))
            nets.append(nw.eval())
        for dicke in nen:
            man = faelle(dicke)
            lader = turnier.mit_manifest([MAN[dicke]])
            lader()
            for number, case in enumerate(man, 1):
                sid, k = case["id"], case["fold"]
                p = f"{out_dir}/{name}/{dicke}/fold{k}/{sid}_pred_proba.nii.gz"
                if os.path.exists(p):
                    continue
                os.makedirs(os.path.dirname(p), exist_ok=True)
                proba, nka = cv5.proba_fall(nets[k], sid, bs=16, nch=2)   # (z,y,x)
                rec = cv5.FA["recs"][sid]
                back = np.ascontiguousarray(proba.transpose(2, 1, 0))
                nib.save(nib.Nifti1Image(np.round(back * 255).astype(np.uint8), rec["affine"]), p)
                print(f"  [{name}/{dicke}] {number}/{len(man)} {sid} kacheln={nka}", flush=True)
    return out


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu")


def auswerten(models, out):
    """Metriken je (modell, dicke, betrieb) + Bootstrap gegen native."""
    rows = {}
    for name, _ in models:
        for dicke in NEN:
            for bet, thrname in (("fest", "0.3"),):
                r = []
                for case in faelle(dicke):
                    sid, k, d = case["id"], case["fold"], case["dir"]
                    pr = np.asarray(nib.load(f"{out}/{name}/{dicke}/fold{k}/{sid}_pred_proba.nii.gz").dataobj, dtype=np.float32) / 255.
                    ref = np.asarray(nib.load(f"{d}/{sid}_label.nii.gz").dataobj) > 0
                    h = hits(ref, post_process(pr, THRESHOLD))
                    aff = nib.load(f"{d}/{sid}_image.nii.gz").affine
                    h["groessen"] = [groessen_klasse(g) for g in komponenten_groesse_mm(ref, aff)]
                    r.append(h)
                rows[(name, dicke, "fest")] = summarize(r)
    # Betrieb mit innerfold-Schwelle (je Falte aus meta.json)
    for name, mpfad in models:
        for dicke in NEN:
            r = []
            for case in faelle(dicke):
                sid, k, dd = case["id"], case["fold"], case["dir"]
                thr = json.load(open(f"{mpfad}/fold{k}/meta.json"))["schwelle"]
                pr = np.asarray(nib.load(f"{out}/{name}/{dicke}/fold{k}/{sid}_pred_proba.nii.gz").dataobj, dtype=np.float32) / 255.
                ref = np.asarray(nib.load(f"{dd}/{sid}_label.nii.gz").dataobj) > 0
                r.append(hits(ref, post_process(pr, thr)))
            rows[(name, dicke, "innerfold")] = summarize(r)
    # Patientenlevel + Bootstrap (fester Betrieb 0.3)
    pat, boot = {}, {}
    for name, _ in models:
        per_case = {}
        for dicke in NEN:
            c = []
            for case in faelle(dicke):
                sid, k, dd = case["id"], case["fold"], case["dir"]
                pr = np.asarray(nib.load(f"{out}/{name}/{dicke}/fold{k}/{sid}_pred_proba.nii.gz").dataobj, dtype=np.float32) / 255.
                ref = np.asarray(nib.load(f"{dd}/{sid}_label.nii.gz").dataobj) > 0
                c.append((sid, hits(ref, post_process(pr, THRESHOLD)),
                          [groessen_klasse(g) for g in komponenten_groesse_mm(ref, nib.load(f"{dd}/{sid}_image.nii.gz").affine)]))
            per_case[dicke] = c
        n_pat = len(per_case["native"])
        for dicke in NEN:
            cnt_r = [h["n_ref"] for _, h, _ in per_case[dicke]]
            cnt_p = [h["n_pred"] for _, h, _ in per_case[dicke]]
            pear = float(scistats.pearsonr(cnt_r, cnt_p)[0]) if np.std(cnt_p) > 0 else float("nan")
            spear = float(scistats.spearmanr(cnt_r, cnt_p)[0]) if np.std(cnt_p) > 0 else float("nan")
            pos = [i for i, (sid, h, g) in enumerate(per_case[dicke]) if h["n_ref"] > 4]
            tp = sum(per_case[dicke][i][1]["tp"] > 0 for i in pos)
            neg = [i for i in range(n_pat) if per_case[dicke][i][1]["n_ref"] <= 4]
            tn = sum(per_case[dicke][i][1]["n_pred"] == 0 for i in neg)
            pat.setdefault(name, {})[dicke] = dict(rank_pearson=round(pear, 3), rank_spearman=round(spear, 3),
                                                   pos_n=len(pos), pos_tp=tp,
                                                   sens_gt4=round(tp / len(pos), 3) if pos else None,
                                                   neg_n=len(neg), tn_gt4=tn, spec_gt4=round(tn / len(neg), 3) if neg else None)
        # Sens nach Groesse je Dicke (pooled) -- wird unten korrekt berechnet (sens_g)
        # gepaarter Bootstrap gegen native: Faelle mit Rueckgabe resampleln,
        # je Iteration tp/fp/fn pooled, Delta der pooled-Metriken.
        rng = np.random.default_rng(42)
        n_iter = 1000
        for dicke in [d for d in ("n3", "n5", "n7") if d in per_case]:
            nat = [h for _, h, _ in per_case["native"]]
            thi = [h for _, h, _ in per_case[dicke]]
            deltas = {"f1": np.empty(n_iter), "sens": np.empty(n_iter), "prec": np.empty(n_iter), "fp_case": np.empty(n_iter)}
            for b in range(n_iter):
                idx = rng.integers(0, n_pat, n_pat)
                a_s = {k: sum(nat[i][k] for i in idx) for k in ("tp", "fp", "fn")}
                c_s = {k: sum(thi[i][k] for i in idx) for k in ("tp", "fp", "fn")}
                def pooled(s, m):
                    if m == "f1":
                        return 2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1)
                    if m == "sens":
                        return s["tp"] / max(s["tp"] + s["fn"], 1)
                    if m == "prec":
                        return s["tp"] / max(s["tp"] + s["fp"], 1)
                    return s["fp"] / len(idx)
                for key in deltas:
                    deltas[key][b] = pooled(c_s, key) - pooled(a_s, key)
                if b % 100 == 0:
                    print(f"  bootstrap {name}/{dicke} {b}/{n_iter}", flush=True)
            boot.setdefault(name, {})[dicke] = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4), round(float(v.mean()), 4)] for k, v in deltas.items()}
    # Sens nach Groesse (richtig, je Referenz-Komponente via masken)
    sens_g = {}
    for name, _ in models:
        for dicke in NEN:
            agg = {k: [0, 0] for k in ("<2", "2-3", "3-5", ">5")}
            for case in faelle(dicke):
                sid, k = case["id"], case["fold"]
                dd = case["dir"]
                pr = np.asarray(nib.load(f"{out}/{name}/{dicke}/fold{k}/{sid}_pred_proba.nii.gz").dataobj, dtype=np.float32) / 255.
                ref = np.asarray(nib.load(f"{dd}/{sid}_label.nii.gz").dataobj) > 0
                pred = post_process(pr, THRESHOLD)
                aff = nib.load(f"{dd}/{sid}_image.nii.gz").affine
                lr, nr = ndimage.label(ref, structure=N26)
                lp, np_ = ndimage.label(pred, structure=N26)
                both = ref & pred
                pred_hit = np.zeros(np_ + 1, bool)
                if both.any():
                    pred_hit[np.unique(lp[both])] = True
                for comp in range(1, nr + 1):
                    g = groessen_klasse(komponenten_groesse_mm(ref * (lr == comp), aff)[0])
                    agg[g][1] += 1
                    hit = (lr == comp) & pred
                    agg[g][0] += int(hit.any())
            sens_g.setdefault(name, {})[dicke] = {k: {"n_ref": v[1], "n_hit": v[0],
                                                      "sens": round(v[0] / v[1], 3) if v[1] else None} for k, v in agg.items()}
    # Nutzerregel: CMB werden in DREI Klassen berichtet (<2 / 2-5 / >5 mm). Der Auftrag
    # verlangt zusaetzlich die feinere 4er-Aufteilung oben; hier dieselben Zaehler zusammengefasst.
    sens_g3 = {}
    for name, je in sens_g.items():
        for dicke, kl in je.items():
            a = kl["<2"]; b = kl["2-3"]; c = kl["3-5"]; d = kl[">5"]
            def _m(x, y):
                n = x["n_ref"] + y["n_ref"]
                h = x["n_hit"] + y["n_hit"]
                return {"n_ref": n, "n_hit": h, "sens": round(h / n, 3) if n else None}
            sens_g3.setdefault(name, {})[dicke] = {"<2": a, "2-5": _m(b, c), ">5": d}
    # dict-Schluessel sind Tupel -> fuer JSON und fuer die Figur eine flache Liste.
    # summarize() liefert "sensitivity"/"fp_per_case", die Figur liest genau diese Namen.
    rows = [dict(modell=n, dicke=d, betrieb=b, **v) for (n, d, b), v in rows.items()]
    return rows, pat, boot, sens_g, sens_g3


out_dir = None
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelle", nargs="+", required=True, help="NAME=PFAD")
    ap.add_argument("--net", default="a03-aniso")
    ap.add_argument("--out", default=f"{ROOT}/ergebnisse/schichtdicke")
    ap.add_argument("--nur-prognose", action="store_true")
    a = ap.parse_args()
    out_dir = a.out
    models = [tuple(x.split("=", 1)) for x in a.modelle]
    t0 = time.time()
    print("Vorhersagen (CPU, out-of-fold):", models, flush=True)
    vorhersagen(models, a.net, NEN)
    print(f"Prognose fertig in {time.time() - t0:.0f} s", flush=True)
    if a.nur_prognose:
        sys.exit(0)
    t1 = time.time()
    rows, pat, boot, sens_g, sens_g3 = auswerten(models, out_dir)
    daten = {"modelle": {n: p for n, p in models}, "net": a.net, "betriebe": rows,
             "patientenlevel": pat, "bootstrap_95ci_gegen_native": boot,
             "sensitivitaet_nach_groesse": sens_g,
             "sensitivitaet_nach_groesse_3klassen": sens_g3,
             "zeit_s": {"prognose": round(time.time() - t0, 1), "auswertung": round(time.time() - t1, 1)},
             "schwellen": {"fest": THRESHOLD, "min_mm3": MIN_MM3, "max_voxel": MAX_VOXELS},
             "n_faelle": 27, "ko": "valdo-2 (RSS 0.8 mm)"}
    # Auftrag Step 6 nennt summary.json; metriken.json bleibt als alter Name daneben.
    for datei in ("summary.json", "metriken.json"):
        json.dump(daten, open(f"{a.out}/{datei}", "w"), indent=1)
        print(f"{datei} geschrieben")
