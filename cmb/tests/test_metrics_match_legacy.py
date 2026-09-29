"""Prueft, dass `cmb.metrics` die Altfassungen `code/metric.py` und `code/metric_valdo.py` EXAKT reproduziert.

Abnahmeregel des Pakets (cmb/README.md): ein Modul gilt erst, wenn es seinen Vorgaenger exakt reproduziert. Hier
heisst das: gleiche Zaehlungen auf Zufallsmasken UND auf echten Daten, und gleiche Zuordnung bei der VALDO-Regel.

    python -m cmb.tests.test_metrics_match_legacy

Rueckgabe 0 wenn alles stimmt, 1 bei der ersten Abweichung (mit dem Fall, der sie ausloest).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/code")

import metric as alt_touch                    # noqa: E402
import metric_valdo as alt_valdo              # noqa: E402
from cmb import metrics as neu                # noqa: E402


def zufallsmasken(rng, form=(40, 50, 50), n=12):
    """Masken mit realistischer Struktur: wenige kleine Klumpen, nicht Rauschen -- sonst zerfaellt alles in
    Tausende Komponenten und der Vergleich prueft nur noch die Zaehlschleife."""
    for _ in range(n):
        a = np.zeros(form, bool); b = np.zeros(form, bool)
        for ziel, k in ((a, rng.integers(0, 6)), (b, rng.integers(0, 8))):
            for _ in range(k):
                z, y, x = (rng.integers(3, s - 4) for s in form)
                r = int(rng.integers(1, 3))
                ziel[z - r:z + r + 1, y - r:y + r + 1, x - r:x + r + 1] = True
        yield a, b


def f1_gleich(alt, neu_wert) -> bool:
    """Vergleicht f1 tolerant gegen zwei bewusste Unterschiede: die Altfassung rundet auf vier Stellen, und fuer
    Faelle OHNE Referenz-Blutung liefert sie None statt nan (beide heissen 'nicht definiert')."""
    import math
    undefiniert = lambda v: v is None or (isinstance(v, float) and math.isnan(v))
    if undefiniert(alt) or undefiniert(neu_wert):
        return undefiniert(alt) and undefiniert(neu_wert)
    return round(float(neu_wert), 4) == round(float(alt), 4)


def main() -> int:
    fehler = 0
    rng = np.random.default_rng(0)

    # --- 1) Beruehrungsregel auf Zufallsmasken
    for i, (ref, pred) in enumerate(zufallsmasken(rng, n=40)):
        a, b = alt_touch.hits(ref, pred), neu.hits(ref, pred)
        if a != b:
            print(f"ABWEICHUNG hits, Zufallsfall {i}: alt {a} gegen neu {b}"); fehler += 1; break
    else:
        print("hits auf 40 Zufallsmasken: identisch")

    # --- 2) summarize auf denselben Zeilen
    zeilen = [neu.hits(r, p) for r, p in zufallsmasken(np.random.default_rng(1), n=25)]
    a, b = alt_touch.summarize(zeilen), neu.summarize(zeilen)
    if a != b:
        print(f"ABWEICHUNG summarize: alt {a} gegen neu {b}"); fehler += 1
    else:
        print("summarize auf 25 Faellen: identisch")

    # --- 3) VALDO-Regel: Zuordnung und Zaehlung
    for i, (ref, pred) in enumerate(zufallsmasken(np.random.default_rng(2), n=30)):
        a = alt_valdo.case_valdo(ref, pred.astype(np.float32), (0.5, 0.5, 1.0))
        b = neu.case_valdo(ref, pred.astype(np.float32), (0.5, 0.5, 1.0))
        # Die Altfassung rundet f1 auf vier Stellen und liefert Zusatzfelder (match_dist_mm_max, multi_hit, m26_*),
        # die nur ihr eigenes Kommandozeilenprogramm benutzt. Verglichen werden die gemeinsamen Felder; f1 mit
        # derselben Rundung -- die neue Fassung rundet absichtlich NICHT, das gehoert in die Berichtsschicht.
        gleich = all(a[k] == b[k] for k in ("n_ref", "n_pred", "tp", "fp", "fn", "aed")) and f1_gleich(a["f1"], b["f1"])
        if not gleich:
            print(f"ABWEICHUNG case_valdo, Zufallsfall {i}: alt {a} gegen neu {b}"); fehler += 1; break
    else:
        print("case_valdo auf 30 Zufallsmasken: identisch")

    # --- 4) echte Daten: Referenz gegen eine echte Vorhersage
    gitter, lauf = f"{ROOT}/dev/gitter_d", f"{ROOT}/ergebnisse/turnier/a03-aniso-e60-gd"
    pfade = sorted(glob.glob(f"{lauf}/fold[0-9]/*_pred_proba.nii.gz"))[:8]
    if not pfade:
        print("echte Daten: uebersprungen (keine Vorhersagen vorhanden)")
    else:
        import nibabel as nib
        n_gleich = 0
        for pfad in pfade:
            fall = os.path.basename(pfad)[: -len("_pred_proba.nii.gz")]
            p = np.asarray(nib.load(pfad).dataobj).astype(np.float32) / 255.0
            ref = np.asarray(nib.load(f"{gitter}/{fall}/{fall}_label.nii.gz").dataobj) > 0
            if alt_touch.hits(ref, p > 0.3) != neu.hits(ref, p > 0.3):
                print(f"ABWEICHUNG hits auf echtem Fall {fall}"); fehler += 1; break
            a, b = alt_valdo.case_valdo(ref, p, (0.5, 0.5, 1.0)), neu.case_valdo(ref, p, (0.5, 0.5, 1.0))
            if not all(a[k] == b[k] for k in ("tp", "fp", "fn", "aed")):
                print(f"ABWEICHUNG case_valdo auf echtem Fall {fall}: alt {a} gegen neu {b}"); fehler += 1; break
            n_gleich += 1
        else:
            print(f"echte Daten: {n_gleich} Faelle, beide Regeln identisch")

    print("\nalles identisch" if not fehler else f"\n{fehler} Abweichung(en)")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
