#!/usr/bin/env python3
"""we4 CV5 Aggregation (Neufassung Stand 11.09. Claude; Vorfassung code/archiv/cv5_agg.py.vor-0911).

Ablauf: ERST alles pruefen und rechnen, DANN schreiben -- summary.json (die Datei, auf
die die Abnahme schaut) entsteht als letzte Datei ihres Ordners, jede Datei atomar
(tmp + os.replace). Ein Absturz vorher hinterlaesst keine Abnahme-Datei.

Konfigurationen = Schluessel von KONFS in code/cv5.py (per ast gelesen, ab 11.09. r1/r2/r3;
Rueckfall dev/optuna_top3.json -> t<trial>). Andere ergebnisse/cv5-*-Ordner werden als
'fremd' gemeldet, nicht aggregiert. Falten und Validierung aus dev/split.json (Anzahl
Falten und Faelle NICHT fest verdrahtet). Labels: Gitter aus meta.json 'gitter'
(dev/<gitter>), sonst dev/gitter_c, dev/gitter_b -- immer mit Pruefung shape + affine.

Je Konfiguration:
  - Vollstaendigkeit: je Falte meta.json mit ok=true, modell.pt, fuer JEDEN Fall der
    Testfalte <id>_pred.nii.gz; unvollstaendige Konfigurationen werden mit Grund
    GEMELDET (Log + ergebnisse/cv5_agg_bericht.json), nie still uebersprungen.
  - Integritaet JE KONFIGURATION: jeder Fall genau einmal, nur in seiner Testfalte,
    kein Validierungsfall; Validierung und Falten in split.json disjunkt.
  - Zahlen aus den Vorhersage-DATEIEN neu gezaehlt: code/metric.py (hits/summarize,
    26er-Regel, unveraendert importiert) UND code/metric_valdo.py (offizielle
    VALDO-Regel, 6er, Schwerpunkt < 5 mm, eins-zu-eins; dazu multi_hit und groesste
    Komponente in mm3). Weicht eine Datei von ihrem <id>_pred_meta.json ab, ist die
    Konfiguration inkonsistent (veraltete Vorhersage) und zaehlt nicht als vollstaendig.
  - F1 je Falte als Mittel +- Std; je Kohorte (Ko1/Ko2/Ko3, aus sub-<K>xx) mit HINWEIS
    bei < 5 Faellen; Klumpen (verworfene Riesenkomponenten) aus meta.json.

Sieger = hoechster gepoolter 26er-F1 (Tie: Sensitivitaet, VALDO-F1 gepoolt, Name). ACHTUNG: diese Wahl
faellt auf den TESTFALTEN derselben CV und ist damit leicht optimistisch -- in
summary.json und lauf.json als 'auswahl_auf_testfalten': true gekennzeichnet.

Schrift: ergebnisse/cv5-<konf>/{summary.json,results.csv} (nur vollstaendige),
ergebnisse/cv5-best/{results.csv,auswahl.json,fold<k>/<id>_pred.nii.gz (Links),summary.json},
dev/modelle/best/{fold<k>/modell.pt,fold<k>/meta.json,sha256.txt (sha256sum -c-Format),
lauf.json}, ergebnisse/cv5_agg_bericht.json (immer).
Rueckgabe: 0 ok, 1 keine Konfiguration vollstaendig, 2 Integritaet verletzt
(Validierungsleck, Fall doppelt/falsche Falte), 3 Auswahl/Modelle widerspruechlich.

Aufruf:  python code/cv5_agg.py [--nur-pruefen]   (CPU; CV5_AGG_WORKER=4 Prozesse)
"""
import os, sys, json, glob, csv, io, ast, time, hashlib, shutil, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import nibabel as nib

ROOT = os.environ.get("VALDO_ROOT", os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
from metric import hits, summarize
from metric_valdo import case_valdo, summarize_valdo, label_for

TOP3_PATH = f"{ROOT}/dev/optuna_top3.json"
NIEDRIG = 5              # <5 Faelle je Kohorte -> HINWEIS, nicht Interpretation
WORKER = int(os.environ.get("CV5_AGG_WORKER", "4"))
ERWARTET_META = ("konf", "trial", "falte", "ok", "params", "epochs", "schwelle", "mindestgroesse",
                 "f1_testfalte", "klumpen_testfalte", "max_komp_vox",
                 "rand_versatz", "innere_falte", "epoche_gewaehlt")
HITS_KEYS = ("n_ref", "n_pred", "tp", "fp", "fn", "case_ref", "case_pred", "count_error")
AUSWAHL_HINWEIS = ("Sieger = hoechster gepoolter F1 ueber die Testfalten derselben CV. Die Wahl unter "
                   "den Konfigurationen faellt damit auf den Testfalten und ist leicht optimistisch; "
                   "unverzerrt ist erst die einmalige Validierung (we5).")


# ---------------------------------------------------------------- Hilfen
def kohorte(sid):
    # sub-<K><nn>: Kohorte = 5. Zeichen (sub-101 -> Ko1). Vorfassung las sid[5] -> falsch.
    return f"Ko{sid[4]}" if sid.startswith("sub-") and len(sid) > 4 else None

def lies_json(p):
    try:
        return json.load(open(p)), None
    except FileNotFoundError:
        return None, "fehlt"
    except Exception as e:
        return None, f"unlesbar ({type(e).__name__}: {e})"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def schreibe_atomar(pfad, inhalt):
    os.makedirs(os.path.dirname(pfad), exist_ok=True)
    tmp = f"{pfad}.tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(inhalt if isinstance(inhalt, str) else json.dumps(inhalt, indent=1, ensure_ascii=False))
    os.replace(tmp, pfad)

def kopiere_atomar(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp-{os.getpid()}"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def link_atomar(src, dst):
    tmp = f"{dst}.tmp-{os.getpid()}"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(os.path.abspath(src), tmp)
    os.replace(tmp, dst)

def beiseite(pfad, endung):
    """Alte Abnahme-Datei vor dem Schreiben wegstellen (kein Loeschen)."""
    if os.path.exists(pfad):
        os.replace(pfad, pfad + endung)

def csv_text(rows):
    buf = io.StringIO()
    keys = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(buf, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return buf.getvalue()

def code_konstanten(pfad):
    """Grossgeschriebene Modulkonstanten (Literale) aus cv5.py -- Stand der Datei beim
    Aggregieren, NICHT zwingend beim Training; massgeblich ist meta_je_falte."""
    try:
        baum = ast.parse(open(pfad).read())
    except Exception as e:
        return {"fehler": str(e)}
    out = {}
    for n in baum.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id.isupper():
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass
    return out


# ---------------------------------------------------------------- Konfigurationen
def konf_namen_cv5():
    """Schluessel von KONFS aus code/cv5.py (per ast, ohne torch-Import). None, wenn nicht lesbar."""
    try:
        baum = ast.parse(open(f"{CODE}/cv5.py").read())
    except Exception:
        return None
    for n in baum.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and getattr(n.targets[0], "id", "") == "KONFS" \
                and isinstance(n.value, ast.Dict):
            return [ast.literal_eval(k) for k in n.value.keys]
    return None

def konfigurationen():
    """Massgeblich: KONFS in code/cv5.py (r1/r2/r3 ab 11.09.). Rueckfall: optuna_top3.json (t<trial>).
    Weitere cv5-*-Ordner werden nur als 'fremd' gemeldet, nicht aggregiert."""
    namen = konf_namen_cv5()
    quelle = "code/cv5.py KONFS"
    if not namen:
        top, _ = lies_json(TOP3_PATH)
        namen = [f"t{c['trial']}" for c in (top or []) if isinstance(c, dict) and "trial" in c]
        quelle = "dev/optuna_top3.json (Rueckfall)"
    fremd = [os.path.basename(d)[len("cv5-"):] for d in sorted(glob.glob(f"{ROOT}/ergebnisse/cv5-*"))
             if os.path.isdir(d) and not os.path.basename(d).startswith("cv5-best")
             and os.path.basename(d)[len("cv5-"):] not in namen]
    return {n: None for n in namen}, quelle, fremd

def we3_einfalten_f1():
    """Bester we3-Einfalten-F1 (anderes Rezept, Testfalte 0) aus optuna_top3.json -- Vergleichszahl."""
    top, _ = lies_json(TOP3_PATH)
    f = [c.get("f1") for c in (top or []) if isinstance(c, dict) and isinstance(c.get("f1"), (int, float))]
    return max(f) if f else None

def pruefe_konf(name, folds, val):
    """Vollstaendigkeit + Integritaet EINER Konfiguration (keine Zahlen)."""
    d = f"{ROOT}/ergebnisse/cv5-{name}"
    probleme, hinweise, leck, metas, faelle = [], [], [], {}, []
    if not os.path.isdir(d):
        return dict(vollstaendig=False, probleme=["Ordner ergebnisse/cv5-%s fehlt" % name],
                    hinweise=[], leck=[], metas={}, faelle=[], integritaet={})
    gesehen = {}
    for k, fold_ids in enumerate(folds):
        fd = f"{d}/fold{k}"
        if not os.path.isdir(fd):
            probleme.append(f"fold{k}: Ordner fehlt"); continue
        m, err = lies_json(f"{fd}/meta.json")
        if m is None:
            probleme.append(f"fold{k}: meta.json {err}")
        elif not m.get("ok"):
            probleme.append(f"fold{k}: meta.json ok != true")
        else:
            if m.get("falte") not in (None, k):
                probleme.append(f"fold{k}: meta.json falte={m.get('falte')}")
            if m.get("konf") not in (None, name):
                probleme.append(f"fold{k}: meta.json konf={m.get('konf')}")
            metas[k] = m
        if not os.path.exists(f"{fd}/modell.pt"):
            probleme.append(f"fold{k}: modell.pt fehlt")
        vorh = {os.path.basename(f)[: -len("_pred.nii.gz")] for f in glob.glob(f"{fd}/*_pred.nii.gz")}
        for sid in vorh:
            gesehen.setdefault(sid, []).append(k)
        fehl = sorted(set(fold_ids) - vorh); extra = sorted(vorh - set(fold_ids))
        if fehl:
            probleme.append(f"fold{k}: {len(fehl)} von {len(fold_ids)} Vorhersagen fehlen (z.B. {fehl[:3]})")
        if extra:
            probleme.append(f"fold{k}: {len(extra)} Vorhersagen fuer Faelle ausserhalb dieser Testfalte {extra[:5]}")
        leck += sorted(vorh & set(val))
        ohne_meta = [s for s in sorted(vorh & set(fold_ids)) if not os.path.exists(f"{fd}/{s}_pred_meta.json")]
        if ohne_meta:
            hinweise.append(f"fold{k}: {len(ohne_meta)} ohne _pred_meta.json (kein Abgleich Datei<->Meta)")
        ohne_proba = [s for s in sorted(vorh & set(fold_ids)) if not os.path.exists(f"{fd}/{s}_pred_proba.nii.gz")]
        if ohne_proba:
            hinweise.append(f"fold{k}: {len(ohne_proba)} ohne _pred_proba.nii.gz")
        faelle += [(sid, k) for sid in sorted(vorh & set(fold_ids))]
    doppelt = sorted(s for s, ks in gesehen.items() if len(ks) > 1)
    n_soll = int(sum(len(f) for f in folds))
    integ = dict(
        validierung_nicht_vorhergesagt=not leck,
        jeder_fall_genau_einmal=not doppelt,
        nur_in_eigener_testfalte=not any("ausserhalb dieser Testfalte" in p for p in probleme),
        n_faelle_vorhergesagt=len(gesehen), n_faelle_soll=n_soll,
    )
    if doppelt:
        probleme.append(f"{len(doppelt)} Faelle in mehreren Falten vorhergesagt {doppelt[:5]}")
    return dict(vollstaendig=not probleme, probleme=probleme, hinweise=hinweise, leck=sorted(set(leck)),
                metas=metas, faelle=faelle, integritaet=integ)


# ---------------------------------------------------------------- Zaehlen (CPU, parallel)
def _fall(args):
    sid, k, pp, kandidaten = args
    try:
        pi = nib.load(pp)
        li = label_for(sid, pi, candidates=kandidaten)
        ref = np.asanyarray(li.dataobj) > 0.5
        pred = np.asanyarray(pi.dataobj) > 0.5
        return sid, k, hits(ref, pred), case_valdo(ref, pred, pi.affine), li.get_filename(), None
    except BaseException as e:           # SystemExit aus label_for eingeschlossen
        return sid, k, None, None, None, f"{type(e).__name__}: {e}"

def label_kandidaten(metas):
    kand = []
    for m in metas.values():
        g = m.get("gitter")                       # cv5.py schreibt den Basisnamen, z.B. "gitter_c"
        if isinstance(g, str) and g:
            for gg in ([g] if os.path.isabs(g) else [f"{ROOT}/dev/{g}", f"{ROOT}/{g}"]):
                kand.append(f"{gg}/{{id}}/{{id}}_label.nii.gz")
    kand += [f"{ROOT}/dev/gitter_c/{{id}}/{{id}}_label.nii.gz", f"{ROOT}/dev/gitter_b/{{id}}/{{id}}_label.nii.gz",
             f"{ROOT}/dev/preproc/{{id}}/{{id}}_label.nii.gz"]
    return list(dict.fromkeys(kand))

def zaehle(name, bericht, pool):
    d = f"{ROOT}/ergebnisse/cv5-{name}"
    kand = label_kandidaten(bericht["metas"])
    auftraege = [(sid, k, f"{d}/fold{k}/{sid}_pred.nii.gz", kand) for sid, k in bericht["faelle"]]
    rows, vrows, fehler = [], [], []
    for sid, k, h, v, lab, err in pool.map(_fall, auftraege):
        if err:
            fehler.append(f"fold{k} {sid}: {err}"); continue
        pm, _ = lies_json(f"{d}/fold{k}/{sid}_pred_meta.json")
        if pm is not None:
            abw = {x: (pm[x], h[x]) for x in HITS_KEYS if x in pm and pm[x] != h[x]}
            if abw:
                fehler.append(f"fold{k} {sid}: Datei weicht von _pred_meta.json ab (meta, datei) {abw}")
        m = bericht["metas"].get(k, {})
        basis = dict(id=sid, fold=k, kohorte=kohorte(sid),
                     schwelle=(pm or {}).get("schwelle", m.get("schwelle")),
                     mindestgroesse=(pm or {}).get("mindestgroesse", m.get("mindestgroesse")))
        rows.append(dict(basis, **h))
        vrows.append(dict(basis, **v))
    return rows, vrows, fehler

def zusammenfassung(name, rows, vrows, bericht, folds, top_eintrag, status):
    metas = bericht["metas"]
    z = summarize(rows)
    z["net"] = name
    z["valdo"] = summarize_valdo(vrows)
    z["multi_hit"] = z["valdo"]["multi_hit"]
    z["groesste_komp_mm3"] = z["valdo"]["largest_pred_mm3"]
    z["f1_jede_falte"] = [summarize([r for r in rows if r["fold"] == k])["f1"] for k in range(len(folds))]
    f1s = np.array(z["f1_jede_falte"], float)
    z["f1_falten_mittel"] = round(float(f1s.mean()), 4)
    z["f1_falten_std"] = round(float(f1s.std(ddof=0)), 4)   # KEIN Median (Vertrag)
    z["valdo_f1_gepoolt_jede_falte"] = [summarize_valdo([r for r in vrows if r["fold"] == k])["f1_pooled"]
                                        for k in range(len(folds))]
    koh = {}
    for ko in sorted({r["kohorte"] for r in rows if r["kohorte"]}):
        sub = [r for r in rows if r["kohorte"] == ko]
        s = summarize(sub)
        vs = summarize_valdo([r for r in vrows if r["kohorte"] == ko])
        s["valdo_f1_gepoolt"] = vs["f1_pooled"]; s["valdo_multi_hit"] = vs["multi_hit"]
        s["hinweis"] = "WENIG BESETZT -- nur Hinweis, keine Interpretation" if s["cases"] < NIEDRIG else None
        koh[ko] = s
    z["je_kohorte"] = koh
    kl = [metas[k].get("klumpen_testfalte") for k in sorted(metas)]
    if all(isinstance(x, dict) for x in kl):
        z["klumpen"] = int(sum(x.get("klumpen", 0) for x in kl))
        z["klumpen_voxel"] = int(sum(x.get("klumpen_voxel", 0) for x in kl))
    else:
        z["klumpen"] = None
        z["klumpen_hinweis"] = "klumpen_testfalte fehlt in mindestens einer meta.json"
    z["max_komp_vox"] = sorted({metas[k].get("max_komp_vox") for k in metas}, key=str)
    vr = [metas[k].get("vram_gib") for k in metas if isinstance(metas[k].get("vram_gib"), (int, float))]
    z["vram_gib_max"] = max(vr) if vr else None
    z["dauer_s_je_falte"] = [metas.get(k, {}).get("dauer_s", (status.get(f"{name}/fold{k}") or {}).get("dauer_s"))
                             for k in range(len(folds))]
    we3 = we3_einfalten_f1()
    z["we3_einfalten_f1"] = we3               # bester we3-F1, EINE Falte, anderes Rezept
    z["delta_zu_we3"] = round(z["f1"] - we3, 4) if isinstance(we3, (int, float)) else None
    z["integritaet"] = bericht["integritaet"]
    z["hinweise"] = bericht["hinweise"]
    return z

def auswahl_je_falte(name, metas, n_folds):
    """Gewaehlte Werte je Falte aus meta.json; auswahl.json (falls da) muss dazu passen."""
    aus = {f"fold{k}": {x: metas[k].get(x) for x in ("schwelle", "mindestgroesse", "f1_trainingsfalten",
                                                      "innere_falte", "epoche_gewaehlt") if x in metas[k]}
           for k in range(n_folds)}
    datei, err = lies_json(f"{ROOT}/ergebnisse/cv5-{name}/auswahl.json")
    widerspruch = []
    if datei is not None:
        for fk, w in aus.items():
            for x in ("schwelle", "mindestgroesse"):
                if x in (datei.get(fk) or {}) and datei[fk][x] != w.get(x):
                    widerspruch.append(f"{fk}.{x}: auswahl.json {datei[fk][x]} != meta.json {w.get(x)}")
    return aus, ("auswahl.json" if datei is not None else f"meta.json (auswahl.json {err})"), widerspruch

def lauf_json(name, z, metas, top_eintrag, auswahl, quelle, tabelle, integ_global, modell_sha, n_folds):
    alle = [metas[k] for k in range(n_folds)]
    keys = list(dict.fromkeys(k for m in alle for k in m))
    gemeinsam, je_falte = {}, {}
    for key in keys:
        werte = [m.get(key) for m in alle]
        if all(json.dumps(w, sort_keys=True) == json.dumps(werte[0], sort_keys=True) for w in werte):
            gemeinsam[key] = werte[0]
        else:
            je_falte[key] = werte
    return {
        "aufgabe": "we4 CV5 -- Sieger unter den Konfigurationen (gepoolter CV-F1, 26er-Regel)",
        "stand": "cv5_agg.py Neufassung 11.09. Claude",
        "erzeugt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "konf": name,
        "trial": gemeinsam.get("trial", (top_eintrag or {}).get("trial")),
        "params": gemeinsam.get("params", (top_eintrag or {}).get("params")),
        "auswahl_auf_testfalten": True,
        "auswahl_hinweis": AUSWAHL_HINWEIS,
        "epochs": gemeinsam.get("epochs", je_falte.get("epochs")),
        "max_komp_vox": gemeinsam.get("max_komp_vox", je_falte.get("max_komp_vox")),
        "rand_versatz": gemeinsam.get("rand_versatz", je_falte.get("rand_versatz")),
        "innere_falte": gemeinsam.get("innere_falte", je_falte.get("innere_falte")),
        "epoche_gewaehlt": gemeinsam.get("epoche_gewaehlt", je_falte.get("epoche_gewaehlt")),
        "fehlende_meta_schluessel": [k for k in ERWARTET_META if k not in keys],
        "einstellungen_gemeinsam": gemeinsam,
        "einstellungen_je_falte": je_falte,
        "code_konstanten_cv5_py": code_konstanten(f"{CODE}/cv5.py"),
        "code_konstanten_hinweis": "Stand von code/cv5.py beim Aggregieren, nicht zwingend beim Training; "
                                   "massgeblich sind einstellungen_* aus meta.json",
        "cv5_py_sha256": sha256(f"{CODE}/cv5.py") if os.path.exists(f"{CODE}/cv5.py") else None,
        "split_sha256": sha256(f"{ROOT}/dev/split.json"),
        "auswahl_je_falte": auswahl, "auswahl_quelle": quelle,
        "cv_f1_gepoolt": z["f1"], "sensitivity": z["sensitivity"], "precision": z["precision"],
        "n_pred": z["n_pred"], "cases": z["cases"],
        "f1_jede_falte": z["f1_jede_falte"], "f1_falten_mittel": z["f1_falten_mittel"],
        "f1_falten_std": z["f1_falten_std"],
        "valdo": z["valdo"], "multi_hit": z["multi_hit"], "klumpen": z["klumpen"],
        "groesste_komp_mm3": z["groesste_komp_mm3"],
        "we3_einfalten_f1": z["we3_einfalten_f1"], "delta_zu_we3": z["delta_zu_we3"],
        "konfigurationen": tabelle,
        "integritaet": dict(z["integritaet"], **integ_global),
        "modelle_sha256": modell_sha,
    }


# ---------------------------------------------------------------- Hauptablauf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nur-pruefen", action="store_true", help="alles pruefen und zaehlen, nichts schreiben")
    a = ap.parse_args()
    split, err = lies_json(f"{ROOT}/dev/split.json")
    if split is None:
        print(f"dev/split.json {err}", flush=True); return 1
    folds, val = split["folds"], split["validation"]
    alle_falten = [s for f in folds for s in f]
    integ_global = dict(
        split_validierung_disjunkt_falten=set(val).isdisjoint(alle_falten),
        split_falten_disjunkt=len(alle_falten) == len(set(alle_falten)),
        n_falten=len(folds), n_faelle_falten=len(alle_falten), n_validierung=len(val))
    status, _ = lies_json(f"{ROOT}/dev/cv5_status.json")
    status = status or {}
    konfs, konf_quelle, fremd = konfigurationen()
    # Validierungsleck: alle Vorhersagen der Konfigurationen und von cv5-best (hart);
    # fremde cv5-*-Ordner (altes Rezept/alter Split) nur als Hinweis.
    def vorhergesagt(namen):
        return {os.path.basename(f)[: -len("_pred.nii.gz")] for n in namen
                for f in glob.glob(f"{ROOT}/ergebnisse/cv5-{n}/fold*/*_pred.nii.gz")}
    leck_glob = sorted(vorhergesagt(list(konfs) + ["best"]) & set(val))
    leck_fremd = sorted(vorhergesagt(fremd) & set(val))
    integ_global["validierung_in_keiner_vorhersage"] = not leck_glob

    berichte = {n: pruefe_konf(n, folds, val) for n in konfs}
    tabelle = {n: dict(vollstaendig=b["vollstaendig"], probleme=b["probleme"][:20],
                       n_probleme=len(b["probleme"]), hinweise=b["hinweise"][:10])
               for n, b in berichte.items()}

    def bericht_schreiben(rc, text):
        out = dict(erzeugt=time.strftime("%Y-%m-%d %H:%M:%S"), rc=rc, meldung=text,
                   konf_quelle=konf_quelle, fremde_ordner=fremd, validierungsleck_fremde_ordner=leck_fremd,
                   integritaet_global=integ_global, validierungsleck=leck_glob, konfigurationen=tabelle)
        print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
        if not a.nur_pruefen:
            schreibe_atomar(f"{ROOT}/ergebnisse/cv5_agg_bericht.json", out)
        return rc

    for n, b in berichte.items():
        if not b["vollstaendig"]:
            print(f"[{n}] UNVOLLSTAENDIG: " + "; ".join(b["probleme"][:6]), flush=True)
    if not (integ_global["split_validierung_disjunkt_falten"] and integ_global["split_falten_disjunkt"]):
        return bericht_schreiben(2, "split.json verletzt: Validierung/Falten nicht disjunkt")
    if leck_glob or any(b["leck"] for b in berichte.values()):
        return bericht_schreiben(2, f"VALIDIERUNGSLECK: {leck_glob[:10]}")
    if any(not b["integritaet"].get("jeder_fall_genau_einmal", True) or
           not b["integritaet"].get("nur_in_eigener_testfalte", True) for b in berichte.values()):
        return bericht_schreiben(2, "Fall mehrfach oder in falscher Falte (je Konfiguration, siehe probleme)")
    if not konfs:
        return bericht_schreiben(1, "keine Konfiguration gefunden")
    fertig = [n for n, b in berichte.items() if b["vollstaendig"]]
    if not fertig:
        return bericht_schreiben(1, "Keine Konfiguration vollstaendig -- keine Aggregation")

    # --- Zaehlen (nur vollstaendige), Abgleich Datei <-> _pred_meta.json
    zsums, rows_je = {}, {}
    with ProcessPoolExecutor(max_workers=max(1, WORKER)) as pool:
        for n in fertig:
            rows, vrows, fehler = zaehle(n, berichte[n], pool)
            if fehler:
                berichte[n]["vollstaendig"] = False
                tabelle[n].update(vollstaendig=False, probleme=(fehler + tabelle[n]["probleme"])[:20],
                                  n_probleme=tabelle[n]["n_probleme"] + len(fehler))
                print(f"[{n}] INKONSISTENT: " + "; ".join(fehler[:4]), flush=True)
                continue
            zsums[n] = zusammenfassung(n, rows, vrows, berichte[n], folds, konfs.get(n), status)
            rows_je[n] = (rows, vrows)
    for n, z in zsums.items():
        tabelle[n].update(f1=z["f1"], sensitivity=z["sensitivity"], precision=z["precision"], n_pred=z["n_pred"],
                          cases=z["cases"], f1_falten_mittel=z["f1_falten_mittel"], f1_falten_std=z["f1_falten_std"],
                          valdo_f1_gepoolt=z["valdo"]["f1_pooled"], valdo_f1_fall_median=z["valdo"]["f1_case_median"],
                          multi_hit=z["multi_hit"], klumpen=z["klumpen"], delta_zu_we3=z["delta_zu_we3"])
    if not zsums:
        return bericht_schreiben(1, "Keine Konfiguration vollstaendig und konsistent -- keine Aggregation")

    # --- Sieger + alles, was der Sieger braucht, VOR dem Schreiben pruefen
    # Tie-Break (nur bei exakt gleichem 26er-F1 und gleicher Sensitivitaet): VALDO-F1 gepoolt,
    # dann Name -- sonst gewinnt bei Gleichstand ein Klumpen-Netz ueber den Namen (Test 11.09.).
    name_best = max(zsums, key=lambda n: (zsums[n]["f1"], zsums[n]["sensitivity"],
                                          zsums[n]["valdo"]["f1_pooled"], n))
    metas = berichte[name_best]["metas"]
    auswahl, quelle, widerspruch = auswahl_je_falte(name_best, metas, len(folds))
    if widerspruch:
        return bericht_schreiben(3, f"[{name_best}] auswahl.json widerspricht meta.json: {widerspruch[:4]}")
    modell_sha = {f"fold{k}": sha256(f"{ROOT}/ergebnisse/cv5-{name_best}/fold{k}/modell.pt") for k in range(len(folds))}
    for n in zsums:
        tabelle[n]["sieger"] = n == name_best
    z = dict(zsums[name_best])
    z.update(sieger=True, auswahl_auf_testfalten=True, auswahl_hinweis=AUSWAHL_HINWEIS,
             konfigurationen=tabelle, integritaet=dict(z["integritaet"], **integ_global))
    lauf = lauf_json(name_best, z, metas, konfs.get(name_best), auswahl, quelle, tabelle, integ_global,
                     modell_sha, len(folds))
    lauf["konf_quelle"] = konf_quelle
    if a.nur_pruefen:
        print(json.dumps({"sieger": name_best, "summary": z}, ensure_ascii=False, indent=1)[:20000], flush=True)
        return bericht_schreiben(0, f"nur geprueft, nichts geschrieben; Sieger {name_best}")

    # --- SCHREIBEN (erst jetzt). Je Ordner die Abnahme-Datei zuletzt.
    for n in konfs:
        if n not in zsums and os.path.exists(f"{ROOT}/ergebnisse/cv5-{n}/summary.json"):
            beiseite(f"{ROOT}/ergebnisse/cv5-{n}/summary.json", ".veraltet")   # nicht mehr gueltig
    for n, zz in zsums.items():
        rows, vrows = rows_je[n]
        vcols = [dict(r, **{f"valdo_{k}": v[k] for k in v if k not in r}) for r, v in zip(rows, vrows)]
        schreibe_atomar(f"{ROOT}/ergebnisse/cv5-{n}/results.csv", csv_text(vcols))
        schreibe_atomar(f"{ROOT}/ergebnisse/cv5-{n}/summary.json", zz)

    bd = f"{ROOT}/ergebnisse/cv5-best"
    os.makedirs(bd, exist_ok=True)
    beiseite(f"{bd}/summary.json", ".vorher")
    kopiere_atomar(f"{ROOT}/ergebnisse/cv5-{name_best}/results.csv", f"{bd}/results.csv")
    schreibe_atomar(f"{bd}/auswahl.json", dict(auswahl, quelle=quelle, konf=name_best))
    for k, fold_ids in enumerate(folds):
        kd = f"{bd}/fold{k}"
        os.makedirs(kd, exist_ok=True)
        for f in glob.glob(f"{kd}/*_pred.nii.gz"):          # alte LINKS fremder Faelle weg (nur Links)
            if os.path.islink(f) and os.path.basename(f)[: -len("_pred.nii.gz")] not in fold_ids:
                os.remove(f)
        for sid in fold_ids:
            link_atomar(f"{ROOT}/ergebnisse/cv5-{name_best}/fold{k}/{sid}_pred.nii.gz", f"{kd}/{sid}_pred.nii.gz")
    for f in glob.glob(f"{bd}/fold*"):                        # Falten, die es im Split nicht mehr gibt
        if os.path.isdir(f) and not f.endswith(tuple(f"fold{k}" for k in range(len(folds)))):
            for l in glob.glob(f"{f}/*_pred.nii.gz"):
                if os.path.islink(l):
                    os.remove(l)
    schreibe_atomar(f"{bd}/summary.json", z)

    md = f"{ROOT}/dev/modelle/best"
    os.makedirs(md, exist_ok=True)
    beiseite(f"{md}/lauf.json", ".vorher")
    sha_zeilen = []
    for k in range(len(folds)):
        dst = f"{md}/fold{k}/modell.pt"
        kopiere_atomar(f"{ROOT}/ergebnisse/cv5-{name_best}/fold{k}/modell.pt", dst)
        h = sha256(dst)
        if h != modell_sha[f"fold{k}"]:
            return bericht_schreiben(3, f"Kopie fold{k}/modell.pt weicht vom Original ab")
        sha_zeilen.append(f"{h}  fold{k}/modell.pt")         # sha256sum -c-Format
        schreibe_atomar(f"{md}/fold{k}/meta.json", metas[k])
    schreibe_atomar(f"{md}/sha256.txt", "\n".join(sha_zeilen) + "\n")
    schreibe_atomar(f"{md}/lauf.json", lauf)
    return bericht_schreiben(0, f"Sieger {name_best}: CV-F1 {z['f1']} (auf Testfalten gewaehlt), "
                                f"VALDO-F1 gepoolt {z['valdo']['f1_pooled']}, "
                                f"{sum(1 for t in tabelle.values() if t['vollstaendig'])}/{len(tabelle)} vollstaendig")

if __name__ == "__main__":
    sys.exit(main())
