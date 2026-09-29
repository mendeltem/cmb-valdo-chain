#!/usr/bin/env python3
"""we1: split, stats, cases bauen.

Stratifizierung: Kohorte (Praefix sub-1/2/3) x DREI Lastklassen
(0 / 1-2 / >=3 CMB), NICHT vier -- bei vier sind 3 von 12 Zellen leer.

Validierung: 15 Faelle (~20 %), je Kohorte ein Teil von ~20 % (Ko1 3, Ko2 7,
Ko3 5), jede Kohorte und jede besetzte Zelle vertreten, Faelle nach CMB-Last.
Kohorte 1 MUSS drin sein (15 % der Faelle, 45 % aller Blutungen).
Bis we5 wird sie nicht angefasst.

Falten: 5, gleiche Stratifizierung -- jede Kohorte x Klasse-Zelle wird so
gleich wie moeglich ueber die 5 Falten verteilt (Restfaelle landen in
Falten 0..Rest-1). Innerhalb der Zelle: CMB-Summe der Falte gierig
ausgleichen (naechster Fall in die Falte mit der kleinsten CMB-Summe).
Seed 0. Bei Fehlbelegung: Tausch, kein anderer Seed.

Kein Seed-Knopf fuer die Ausgewogenheit: die Kohorte 3 hat nur 34 CMB in
27 Faellen (26 mit 1-2), und Kohorte 1 hat nur ZWEI Faelle in Klasse 1
(1-2: sub-107 mit 2, sub-102 mit 1 -- die fruehere Zeile "KEINE Falle in
Klasse 1" war falsch, korrigiert 11.09.). Eine Falte ohne positiven
Kohorte-1-Fall laesst sich nicht vermeiden, wenn Kohorte 1 nur 8 positive
Faelle hat und 5 Falten existieren; der gierige Ausgleich stellt sicher,
dass JEDE Falte mindestens einen Kohorte-1-Fall und mindestens einen
positiven Fall enthaelt.

STAND 11.09.2026 (Nutzerentscheidung, Claude) -- DER GUELTIGE SPLIT IST
DER ALTE SEED-0-STAND, genau das, was dieser Code erzeugt:
- Der Text "hoechste CMB zuerst" (oben, im Code-Kommentar und in "regel"
  in split.json) wurde NIE umgesetzt: der Code sortiert nicht, die
  Reihenfolge ist allein die Seed-0-Mischung je Kohorte, die Falten
  bekommen die Restfaelle in Inventar-Reihenfolge. Der Text bleibt in
  "regel" stehen, weil split.json byteidentisch bleiben muss.
- Ergebnis: Validierung 15 Faelle / 97 von 236 CMB (Ko1: sub-109/0,
  sub-102/1, sub-110/73 -> 74 der 106 Ko1-Blutungen), Falten
  [32, 26, 39, 18, 24] CMB. Die Validierung wurde nie vorhergesagt.
- Ein Neusplit wurde am 11.09. gebaut und VERWORFEN: woertlich
  "hoechste zuerst" haette 166/236 CMB in die Validierung gelegt; die
  repraesentative Variante (je Zelle Median-Rang, Falten LPT) 37 CMB --
  aber 13 ihrer 15 Validierungsfaelle lagen im alten Split in den Falten,
  waren also in we2/we3 (Optuna, Lernkurven, cv5) schon trainiert/getestet.
  Die alte Validierung ist die einzige nie angefasste Menge.
  Verworfene Staende: dev/{split,cases,stats}.neusplit-verworfen-0911.json,
  dev/split.py.neusplit-verworfen-0911. Diese Datei ist wieder der Code
  des gueltigen Splits (dev/split.py.vor-neusplit-0911), nur Kommentare
  korrigiert.
"""
import json, random

ROOT = "/home/uchralt/data/work/valdo-t2s"
VALDO = "/home/uchralt/data/extern/valdo2021/Task2"
PREPROC = f"{ROOT}/dev/preproc"

def klass(n):
    return 0 if n == 0 else (1 if n <= 2 else 2)

def main():
    inv = json.load(open(f"{ROOT}/dev/inventar.json"))
    rng = random.Random(0)
    byc = {}
    for c in inv:
        byc.setdefault(c["kohorte"], []).append(c)
    for k in byc:
        rng.shuffle(byc[k])
    assert sum(len(v) for v in byc.values()) == 72
    tot_cmb = sum(c["n_cmb"] for c in inv)

    # --- Validierung: ~20 % je Kohorte (Ko1 3, Ko2 7, Ko3 5 = 15),
    #     jede Kohorte und jede besetzte Zelle vertreten. (Der fruehere Text
    #     "hoechste Last zuerst" wurde nie umgesetzt -- es gilt die
    #     Seed-0-Reihenfolge, siehe STAND 11.09. im Docstring.)
    val_target = {"1": 3, "2": 7, "3": 5}
    val = []
    used = set()
    for koh, t in sorted(val_target.items()):
        cells = {}
        for c in byc[koh]:
            cells.setdefault(klass(c["n_cmb"]), []).append(c)
        # eine Runde je besetzte Zelle (je 1 Fall, der erste in Seed-0-Reihenfolge)
        for kl in sorted(cells):
            c = cells[kl][0]
            val.append(c); used.add(c["id"])
        # Rest bis zum Ziel, in Seed-0-Reihenfolge (nicht nach Last sortiert)
        rest = [c for c in byc[koh] if c["id"] not in used]
        for c in rest:
            if sum(1 for v in val if v["kohorte"] == koh) >= t:
                break
            val.append(c); used.add(c["id"])
    assert len(val) == 15, f"validation {len(val)}"
    assert all(any(v["kohorte"] == k for v in val) for k in ("1", "2", "3"))

    rest = [c for c in inv if c["id"] not in used]
    assert len(rest) == 57

    # --- Falten: jede Kohorte x Klasse-Zelle so gleich wie moeglich ueber
    #     die 5 Falten (Kappe ceil(n/5) je Falte), innerhalb der Zelle
    #     gierig nach CMB-Summe: naechster Fall in die Falte mit der
    #     kleinsten CMB-Summe unter der Kappe. Zellen in fixer Reihenfolge,
    #     damit kleine Zellen (Kohorte 1) zuerst verteilt werden, wenn
    #     alle Summen gleich sind -- dann bekommt jede Falte einen Fall.
    folds = [[] for _ in range(5)]
    cells = {}
    for c in rest:
        cells.setdefault((c["kohorte"], klass(c["n_cmb"])), []).append(c)
    import math
    for zell in sorted(cells):
        zellk, zelll = zell
        lst = cells[zell]
        # Gierig pro Zelle: naechster Fall in die Falte mit der KLEINSTEN
        # Anzahl an Faellen dieser Zelle (dadurch steht jede Zelle in
        # allen/moeglichst vielen Falten), dann kleinste CMB-Summe (der
        # geforderte CMB-Ausgleich), dann Faltenindex.
        for c in lst:
            def key(i, zk=zellk, zl=zelll):
                n_cell = len([x for x in folds[i] if x["kohorte"] == zk
                              and klass(x["n_cmb"]) == zl])
                return (n_cell, sum(x["n_cmb"] for x in folds[i]), i)
            folds[sorted(range(5), key=key)[0]].append(c)
    assert sum(len(f) for f in folds) == 57

    # --- Kontrolle aus Aufgabe 4: jede Falte hat Kohorte-1-Fall UND
    #     positiven Fall. Bei Verstoß: Tausch (kein anderer Seed).
    ok = all(any(c["kohorte"] == "1" for c in f) and any(c["n_cmb"] > 0 for c in f) for f in folds)
    if not ok:
        for i, f in enumerate(folds):
            if not any(c["kohorte"] == "1" for c in f):
                # Kohorte-1-Fall aus einer anderen Falte holen: tauschen mit
                # dem kohorte-1-Fall der Falte mit dem hoechsten CMB-Abstand
                # zum eigenen Mittel, CMB-Summe so nah wie moeglich halten.
                donor = max(range(5), key=lambda j: sum(c["n_cmb"] for c in folds[j]))
                c1 = next(c for c in folds[donor] if c["kohorte"] == "1")
                cswap = min(f, key=lambda c: abs(c["n_cmb"] - c1["n_cmb"]))
                folds[donor][folds[donor].index(c1)] = cswap
                f[f.index(cswap)] = c1
            if not any(c["n_cmb"] > 0 for c in f):
                donor = max((j for j in range(5) if any(c["n_cmb"] > 0 for c in folds[j])),
                            key=lambda j: sum(c["n_cmb"] for c in folds[j]))
                pos = max(folds[donor], key=lambda c: c["n_cmb"])
                cswap = min(f, key=lambda c: abs(c["n_cmb"] - pos["n_cmb"]))
                folds[donor][folds[donor].index(pos)] = cswap
                f[f.index(cswap)] = pos
        ok = all(any(c["kohorte"] == "1" for c in f) and any(c["n_cmb"] > 0 for c in f) for f in folds)
        assert ok, "Faltenkorrektur durch Tausch nicht moeglich"

    # --- split.json
    ids = lambda cs: [c["id"] for c in cs]
    split = {
        "validation": ids(val),
        "folds": [ids(f) for f in folds],
        "regel": ("Stratifiziert nach Kohorte x 3 Lastklassen (0 / 1-2 / >=3 CMB); "
                  "Validierung 15 Faelle (Ko1 3, Ko2 7, Ko3 5), jede Kohorte und je besetzte "
                  "Zelle vertreten, hoechste CMB-Last zuerst; 5 Falten, jede Kohorte x Klasse-Zelle "
                  "gleich verteilt, CMB-Summe je Falte gierig ausgeglichen; Seed 0; "
                  "Validierung bis we5 nicht angefasst."),
        "seed": 0,
    }
    json.dump(split, open(f"{ROOT}/dev/split.json", "w"), indent=1, ensure_ascii=False)

    # --- stats.json: n und CMB-Summe je Kohorte x Klasse x Set/Falte
    def agg(cs):
        d = {}
        for c in cs:
            key = f"ko{c['kohorte']}_kl{klass(c['n_cmb'])}"
            e = d.setdefault(key, {"n": 0, "cmb": 0})
            e["n"] += 1; e["cmb"] += c["n_cmb"]
        return d
    stats = {"gesamt": {"n": 72, "cmb": tot_cmb}, "sets": {}}
    stats["sets"]["validation"] = agg(val)
    for i, f in enumerate(folds):
        stats["sets"][f"fold{i}"] = agg(f)
    stats["kontrolle"] = {
        "validation": {"n": len(val), "cmb": sum(c["n_cmb"] for c in val),
                       "ko1_n": sum(c["kohorte"] == "1" for c in val),
                       "ko1_cmb": sum(c["n_cmb"] for c in val if c["kohorte"] == "1"),
                       "pos_n": sum(c["n_cmb"] > 0 for c in val)},
        "falten": {f"fold{i}": {"n": len(f), "cmb": sum(c["n_cmb"] for c in f),
                                 "ko1_n": sum(c["kohorte"] == "1" for c in f),
                                 "pos_n": sum(c["n_cmb"] > 0 for c in f),
                                 "ko": {k: sum(c["kohorte"] == k for c in f) for k in ("1", "2", "3")}}
                   for i, f in enumerate(folds)},
        "jede_falte_ko1_und_positiv": ok,
    }
    json.dump(stats, open(f"{ROOT}/dev/stats.json", "w"), indent=1, ensure_ascii=False)

    # --- cases.json in der Struktur von code/metric.py:
    #     {"t2star": {"cases": [{"id","fold","klass","mask","image","frst"},...]}}
    fold_of = {}
    for i, f in enumerate(folds):
        for c in f:
            fold_of[c["id"]] = i
    for c in val:
        fold_of[c["id"]] = -1
    cases = []
    for c in inv:
        cid = c["id"]
        cases.append({
            "id": cid,
            "fold": fold_of[cid],
            "klass": klass(c["n_cmb"]),
            "mask": f"{VALDO}/{cid}/{cid}_space-T2S_CMB.nii.gz",
            "image": f"{PREPROC}/{cid}/{cid}_image.nii.gz",
            "frst": f"{PREPROC}/{cid}/{cid}_frst.nii.gz",
        })
    json.dump({"t2star": {"cases": cases}}, open(f"{ROOT}/dev/cases.json", "w"), indent=1, ensure_ascii=False)

    print("validation:", len(val), "cases, CMB", sum(c["n_cmb"] for c in val),
          "| ko1:", sum(c["kohorte"] == "1" for c in val),
          "CMB", sum(c["n_cmb"] for c in val if c["kohorte"] == "1"))
    for i, f in enumerate(folds):
        print(f"fold{i}: n={len(f)} cmb={sum(c['n_cmb'] for c in f)} "
              + " ".join(f"ko{k}={sum(c['kohorte'] == k for c in f)}" for k in ("1", "2", "3")))
    print("kontrolle ko1+positiv je Falte:", ok)

if __name__ == "__main__":
    main()
