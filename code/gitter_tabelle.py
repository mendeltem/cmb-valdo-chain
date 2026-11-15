#!/usr/bin/env python3
"""we2 Gitter: Tabelle A gegen B aus den beiden summary.json + lauf.json.
Schreibt <ROOT>/ergebnisse/gitter_tabelle.md
(gesamt + je Kohorte, F1/Praez/Sens, FP je Fall, Dauer, VRAM-Spitze).
Kein Median -- FP je Fall als Mittel+/-Std, pooled-Pruefgruen aus metric.py.
"""
import json
ROOT = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(v):
    s = json.load(open(f"{ROOT}/ergebnisse/gitter-{v}/summary.json"))
    l = json.load(open(f"{ROOT}/ergebnisse/gitter-{v}/lauf.json"))
    return s, l


def fmt(d):
    if d is None or d.get("n", 0) == 0:
        return "n=0"
    return (f"n={d['n']} F1 {d.get('f1', 0):.3f} | Praez {d.get('praezision', d.get('precision', 0)):.3f} "
            f"| Sens {d.get('sensitivitaet', d.get('sensitivity', 0)):.3f} "
            f"| FP/Fall {d.get('fp_per_case_mean_std')}")


def block(title, a, b):
    out = [f"### {title}", "", "| | A nativ | B gemeinsam |", "|---|---|---|"]
    for key, lab in (("roh", "roh"), ("gefiltert", "gefiltert")):
        out.append(f"| {lab} | {fmt(a[key])} | {fmt(b[key])} |")
    return "\n".join(out)


def main():
    sa, la = load("nativ")
    sb, lb = load("gemeinsam")
    t = []
    t.append("## we2 Gitter -- Tabelle A gegen B (Falte 0, 13 Faelle)")
    t.append("")
    t.append("A nativ: Originalgitter je Fall, Patch 96x64x48 (x*y*z). B gemeinsam: alle Faelle auf "
             "0.5 x 0.5 x 1.0 mm, Patch 94x62x38 (x*y*z) = 98.4 % des physischen Volumens von A in Kohorte 2.")
    t.append("Grossenwahl: 25. Perzentil der kleinsten Referenzkomponente je Trainingsfall (Falten 1-4), "
             f"Kandidaten 0/2/4/8/16 mm3 -> gewaehlt {sa['groessenwahl_mm3']} mm3 (identisch fuer A und B, "
             "weil die Schwelle in mm3 gitterunabhaengig ist).")
    t.append("")
    t.append(block("Gesamt (13 Faelle, pooled)", sa["gesamt"], sb["gesamt"]))
    t.append("")
    for k in ("Ko1", "Ko2", "Ko3"):
        t.append(block(f"{k}", sa["je_kohorte"][k], sb["je_kohorte"][k]))
        t.append("")
    t.append("| | A nativ | B gemeinsam |")
    t.append("|---|---|---|")
    t.append(f"| Trainingsdauer | {la['dauer_s']/60:.1f} min ({la['epochen']} ep) | {lb['dauer_s']/60:.1f} min ({lb['epochen']} ep) |")
    t.append(f"| VRAM-Spitze | {la['vram_spitze_mib']} MiB | {lb['vram_spitze_mib']} MiB |")
    t.append(f"| val_dice best | {la['val_dice_best']} | {lb['val_dice_best']} |")
    t.append(f"| Karten | {la['n_karten']} | {lb['n_karten']} |")
    txt = "\n".join(t) + "\n"
    open(f"{ROOT}/ergebnisse/gitter_tabelle.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
