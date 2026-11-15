"""wc5: sammelt je Arm raw/shrunk/simLo/F1 (je Kohorte) fuer den Bericht."""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json

MB = _P
FIN = json.load(open(f"{MB}/stage2/tset_summary_final.json"))
ARMS = ["tset-tta8", "tset-wide-tta8", "tset-d4", "tset-ctx", "tset-set",
        "tset-proto", "tset-d4proto", "tset-track", "tset-full", "tset-plain"]

rows = []
for arm in ARMS:
    fin = FIN["results"].get(f"pooled {arm}")
    # F1 aus Run-JSONs (alle Seeds)
    seeds_f1 = {}
    for s in (fin["seeds"] if fin else [0]):
        d = json.load(open(f"{MB}/stage2/tset_{arm}_s{s}.json"))
        for model in ("probability", "probability+csf"):
            if model in d.get("results", {}):
                seeds_f1.setdefault(model, []).append({k: v["f1"] for k, v in d["results"][model].items()})
    rows.append((arm, fin, seeds_f1))

# welche Modellzeile passt zur Tabelle in ARBEIT.md (tset-tta8: 0.616/0.670/0.666)?
for model, lst in rows[0][2].items():
    print(model, [f"{v['valdo']:.3f}/{v['t2star']:.3f}/{v['swi']:.3f}" for v in lst])
print()
for arm, fin, sf in rows:
    if fin is None:
        f1 = sf["probability+csf"][-1]
        print(f"{arm}: (Kontrolle) F1+csf valdo={f1['valdo']:.3f} t2star={f1['t2star']:.3f} swi={f1['swi']:.3f}")
        continue
    f1p = sf["probability"][-1]; f1c = sf["probability+csf"][-1]
    print(f"{arm}: seeds={fin['seeds']} raw={fin['delta_all']:+.4f} shrunk={fin['shrunk_delta']:+.4f} "
          f"simLo={fin['simultaneous_lower']:+.4f} | F1 p: {f1p['valdo']:.3f}/{f1p['t2star']:.3f}/{f1p['swi']:.3f} "
          f"| F1+csf: {f1c['valdo']:.3f}/{f1c['t2star']:.3f}/{f1c['swi']:.3f} | {fin['decision']}")
