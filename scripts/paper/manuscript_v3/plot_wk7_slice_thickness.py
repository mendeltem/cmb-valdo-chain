#!/usr/bin/env python3
"""wk7 (07.11.2026): FigS5_slice_thickness.png -- CMB detection under SIMULATED slice thickness.

Kontrolliertes Experiment (public VALDO, RSS, 27 Faelle, native 0.8 mm, eingefrorene Falten):
dasselbe Bild wird n = 3/5/7 Schichten block-averaged (~2.4/4.0/5.6 mm); jede Kohorte 2
ausgewaertete Modelle (HPC-Seeds s42, s43, Thickness-Augmentation) plus das Basis-Modell
a03-aniso-e60-gd (workstation) sagen ihre EIGENE Testfalte bei jeder Dicke aus. Feste
Nachbearbeitung, keine je-Dicke-Schwelle.

    python plot_wk7_slice_thickness.py   (lies ergebnisse/schichtdicke/summary.json)

A  Läsions-F1 und Sensitivität gegen die Schichtdicke (3 Modelle).
B  Sensitivität nach Referenzgröße (<2 / 2-5 / >5 mm, 3 Klassen) gegen Dicke -- hier zeigt
   sich, wo der Einbruch sitzt.
Weisser Hintergrund, PNG.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY = f"{V}/ergebnisse/schichtdicke/summary.json"
OUT = f"{V}/paper/figures/supplement/FigS5_slice_thickness.png"

DICKE = [("native", "0.8"), ("n3", "2.4"), ("n5", "4.0"), ("n7", "5.6")]
MODELL = {
    "base": ("base (workstation)", "#34495e"),
    "s42":  ("s42 (HPC)",    "#e67e22"),
    "s43":  ("s43 (HPC)",    "#27ae60"),
}
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

d = json.load(open(SUMMARY))
rows = d["betriebe"]                      # rows: je Modell+Dicke {f1, sens, prec, fp_scan, ...}
sens3 = d.get("sensitivitaet_nach_groesse_3klassen", {})


def wer(name, dicke, key):
    for r in rows:
        if r["modell"] == name and r["dicke"] == dicke:
            return r[key]
    return None


fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
for name, (lbl, c) in MODELL.items():
    x = range(len(DICKE))
    f1 = [wer(name, dk, "f1") for dk, _ in DICKE]
    se = [wer(name, dk, "sensitivity") for dk, _ in DICKE]
    ax[0].plot(x, f1, "o-", color=c, label=lbl, lw=1.6)
    ax[0].plot(x, se, "s--", color=c, alpha=0.55, lw=1.4)
ax[0].set_xticks(range(len(DICKE)))
ax[0].set_xticklabels([f"{mm} mm" for _, mm in DICKE])
ax[0].set_xlabel("slice thickness (native RSS 0.8 mm)")
ax[0].set_ylabel("lesion F1 / sensitivity")
ax[0].set_ylim(0, 1)
ax[0].legend(title="solid: F1, dashed: sens")
ax[0].grid(alpha=0.25)
ax[0].set_title("A  F1 and sensitivity vs thickness")

for name, (lbl, c) in MODELL.items():
    for gi, klass in enumerate(("<2", "2-5", ">5")):
        ys = [sens3.get(name, {}).get(dk, {}).get(klass, {}).get("sens") for dk, _ in DICKE]
        ax[1].plot(range(len(DICKE)), ys, "o-" if gi == 0 else "s-" if gi == 1 else "D-",
                   color=c, alpha=1.0 if gi == 0 else 0.6 if gi == 1 else 0.35,
                   label=f"{lbl} <2" if gi == 0 else (f"{lbl} 2-5" if gi == 1 else f"{lbl} >5"))
ax[1].set_xticks(range(len(DICKE)))
ax[1].set_xticklabels([f"{mm} mm" for _, mm in DICKE])
ax[1].set_xlabel("slice thickness")
ax[1].set_ylabel("sensitivity by reference size")
ax[1].set_ylim(0, 1.05)
ax[1].grid(alpha=0.25)
ax[1].set_title("B  sensitivity by CMB size (3 classes)")

fig.tight_layout()
fig.savefig(OUT, facecolor="white")
print(f"OK {OUT}")
