#!/usr/bin/env python3
"""Rangliste des Architektur-Turniers (Claude 18.09.2026).

Liest ergebnisse/turnier/<netz>/fold<k>/<id>_pred_meta.json (+ <id>_pred.nii.gz fuer den Dice)
und schreibt ergebnisse/turnier/rangliste.json und rangliste.md. Nur Summen, keine Einzelfaelle.

Je Netz:
  F1 gepoolt ueber alle 57 Faelle (metric.summarize, 26er-Komponenten) -- das RANGMASS.
  Precision, Sensitivitaet, FP je Fall, F1 je Falte (Streuung!).
  Dice: Voxel-Dice gepoolt ueber alle Faelle (2*Summe TP-Voxel / (Summe Ref + Summe Pred)) und
        Mittel je Fall ueber die Faelle MIT Referenz. Bei Objekten von wenigen Voxeln schwankt
        der Dice stark; er ist Nebenmass.
  Gepaart gegen die Messlatte (a01-attunet): Bootstrap ueber FAELLE (10.000 Ziehungen, dieselben
        Faelle fuer beide Netze), Delta des gepoolten F1 mit 95-%-Intervall. Schliesst das
        Intervall die Null ein, ist der Unterschied nicht gesichert -- bei 57 Faellen mit
        Falten-F1 von 0.16 bis 0.49 ist das der Normalfall fuer kleine Unterschiede.
Unvollstaendige Netze (weniger als 5 fertige Falten) stehen in der Liste, aber ohne Rang.

Aufruf:  python turnier_rangliste.py [--basis ergebnisse/turnier] [--messlatte a01-attunet] [--ohne-dice]
"""
import os, sys, json, glob, argparse
import numpy as np

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
from metric import summarize


def f1_aus(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def dice_fall(netz_dir, k, sid, gitter):
    import nibabel as nib
    p = np.asarray(nib.load(f"{netz_dir}/fold{k}/{sid}_pred.nii.gz").dataobj) > 0
    r = np.asarray(nib.load(f"{gitter}/{sid}/{sid}_label.nii.gz").dataobj) > 0
    return int((p & r).sum()), int(r.sum()), int(p.sum())


def liquor_zeile(netz_dir, k, sid, gitter):
    """Liquor-Regel (protokoll.md 18.09.): Komponenten mit Mehrheitslabel Liquor verwerfen, dann hits()."""
    import nibabel as nib
    from scipy import ndimage
    import gewebekarte as G
    from metric import hits
    sp = f"{ROOT}/dev/synthseg/{sid}_synthseg.nii.gz"
    if not os.path.exists(sp):
        return None
    p = np.asarray(nib.load(f"{netz_dir}/fold{k}/{sid}_pred.nii.gz").dataobj) > 0
    r = np.asarray(nib.load(f"{gitter}/{sid}/{sid}_label.nii.gz").dataobj) > 0
    seg = np.asarray(nib.load(sp).dataobj).astype(np.int32)
    lab, n = ndimage.label(p, structure=G.N26)
    if n:
        for c, sl in enumerate(ndimage.find_objects(lab), 1):
            m = lab[sl] == c
            w = seg[sl][m]; w = w[w > 0]
            if len(w) and int(np.bincount(w).argmax()) in G.CSF:
                p[sl][m] = False
    return hits(r, p)


def lies_netz(d, mit_dice, gitter):
    rows, falten = {}, {}
    for k in range(5):
        mp = f"{d}/fold{k}/meta.json"
        if not (os.path.exists(mp) and json.load(open(mp)).get("ok")):
            continue
        falten[k] = json.load(open(mp))
        for f in glob.glob(f"{d}/fold{k}/*_pred_meta.json"):
            r = json.load(open(f))
            if mit_dice:
                r["v_tp"], r["v_ref"], r["v_pred"] = dice_fall(d, k, r["id"], gitter)
                r["liquor"] = liquor_zeile(d, k, r["id"], gitter)
            rows[r["id"]] = r
    return rows, falten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default=f"{ROOT}/ergebnisse/turnier")
    ap.add_argument("--messlatte", default="a01-attunet")
    ap.add_argument("--ohne-dice", action="store_true")
    ap.add_argument("--gitter", default=f"{ROOT}/dev/gitter_c")
    a = ap.parse_args()
    netze = sorted(n for n in os.listdir(a.basis) if os.path.isdir(f"{a.basis}/{n}"))
    daten = {n: lies_netz(f"{a.basis}/{n}", not a.ohne_dice, a.gitter) for n in netze}
    rng = np.random.RandomState(0)
    aus = {}
    for n, (rows, falten) in daten.items():
        if not rows:
            continue
        rl = list(rows.values())
        s = summarize(rl)
        e = dict(falten_fertig=len(falten), n_faelle=len(rl), f1=round(s["f1"], 4),
                 precision=round(s["precision"], 4), sensitivitaet=round(s["sensitivity"], 4),
                 fp_je_fall=round(s["fp_per_case"], 2),
                 f1_je_falte=[falten[k]["f1_testfalte"] if k in falten else None for k in range(5)],
                 epoche_gewaehlt=[falten[k]["epoche_gewaehlt"] if k in falten else None for k in range(5)],
                 parameter_mio=round((falten[min(falten)].get("parameter") or 0) / 1e6, 2),
                 vram_gib=max(f.get("vram_gib") or 0 for f in falten.values()),
                 dauer_h=round(sum(f["dauer_s"] for f in falten.values()) / 3600, 2),
                 bs_abweichend=any(f.get("bs_abweichend") for f in falten.values()),
                 vortrainiert=falten[min(falten)].get("vortrainiert"))
        if not a.ohne_dice:
            vt = sum(r["v_tp"] for r in rl); vr = sum(r["v_ref"] for r in rl); vp = sum(r["v_pred"] for r in rl)
            mit = [2 * r["v_tp"] / (r["v_ref"] + r["v_pred"]) for r in rl if r["v_ref"] > 0]
            e.update(dice_gepoolt=round(2 * vt / (vr + vp), 4) if vr + vp else None,
                     dice_mittel_faelle_mit_ref=round(float(np.mean(mit)), 4) if mit else None,
                     n_faelle_mit_ref=len(mit))
            lq = [r["liquor"] for r in rl if r.get("liquor")]
            if len(lq) == len(rl):
                sl_ = summarize(lq)
                e.update(f1_liquor=round(sl_["f1"], 4), precision_liquor=round(sl_["precision"], 4),
                         sensitivitaet_liquor=round(sl_["sensitivity"], 4), fp_je_fall_liquor=round(sl_["fp_per_case"], 2))
        aus[n] = e
    # gepaart gegen die Messlatte
    if a.messlatte in daten and daten[a.messlatte][0]:
        ref = daten[a.messlatte][0]
        for n, (rows, _) in daten.items():
            if n == a.messlatte or n not in aus:
                continue
            ids = sorted(set(rows) & set(ref))
            if len(ids) < 10:
                continue
            A = [rows[i] for i in ids]; B = [ref[i] for i in ids]
            d = []
            for _ in range(10000):
                ix = rng.randint(len(ids), size=len(ids))
                d.append(f1_aus([A[j] for j in ix]) - f1_aus([B[j] for j in ix]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            if all(x.get("liquor") for x in A + B):
                AL = [x["liquor"] for x in A]; BL = [x["liquor"] for x in B]; dl = []
                for _ in range(10000):
                    ix = rng.randint(len(ids), size=len(ids))
                    dl.append(f1_aus([AL[j] for j in ix]) - f1_aus([BL[j] for j in ix]))
                lo2, hi2 = np.percentile(dl, [2.5, 97.5])
                aus[n]["gegen_messlatte_liquor"] = dict(delta_f1=round(f1_aus(AL) - f1_aus(BL), 4),
                                                        ki95=[round(float(lo2), 4), round(float(hi2), 4)],
                                                        gesichert=bool(lo2 > 0 or hi2 < 0))
            aus[n]["gegen_messlatte"] = dict(n_gemeinsam=len(ids), delta_f1=round(f1_aus(A) - f1_aus(B), 4),
                                             ki95=[round(float(lo), 4), round(float(hi), 4)],
                                             gesichert=bool(lo > 0 or hi < 0))
    rang = sorted([n for n in aus if aus[n]["falten_fertig"] == 5], key=lambda n: -aus[n]["f1"])
    for i, n in enumerate(rang, 1):
        aus[n]["rang"] = i
    json.dump(dict(messlatte=a.messlatte, rang=rang, netze=aus), open(f"{a.basis}/rangliste.json", "w"),
              indent=1, ensure_ascii=False)
    z = ["| Rang | Netz | F1 | F1 +Liquor | Prec | Sens | FP/Fall | Dice gepoolt | F1 je Falte | Delta F1 gegen Messlatte [KI95] | Mio Par. | h |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in rang + [m for m in aus if m not in rang]:
        e = aus[n]; g = e.get("gegen_messlatte")
        gs = f"{g['delta_f1']:+.3f} [{g['ki95'][0]:+.3f}; {g['ki95'][1]:+.3f}]{' *' if g['gesichert'] else ''}" if g else "-"
        fj = " ".join("-" if v is None else f"{v:.2f}" for v in e["f1_je_falte"])
        z.append(f"| {e.get('rang', '(' + str(e['falten_fertig']) + '/5)')} | {n} | {e['f1']:.3f} | {e.get('f1_liquor', '-')} | {e['precision']:.2f} | "
                 f"{e['sensitivitaet']:.2f} | {e['fp_je_fall']:.1f} | {e.get('dice_gepoolt', '-')} | {fj} | {gs} | "
                 f"{e['parameter_mio']} | {e['dauer_h']} |")
    open(f"{a.basis}/rangliste.md", "w", encoding="utf-8").write("\n".join(z) + "\n")
    print("\n".join(z))


if __name__ == "__main__":
    main()
