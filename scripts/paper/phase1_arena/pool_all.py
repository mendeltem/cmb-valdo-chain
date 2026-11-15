import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json, numpy as np, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # sibling scripts (were in the private extern folder)
sys.path.insert(0, _R)
from auswertung import f1_pool, bootstrap_ci

def pool(rows):
    t = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn")}
    n = len(rows)
    return dict(n_cases=n, tp=t["tp"], fp=t["fp"], fn=t["fn"],
                precision=round(t["tp"]/max(t["tp"]+t["fp"],1),4),
                sensitivity=round(t["tp"]/max(t["tp"]+t["fn"],1),4),
                f1=round(2*t["tp"]/max(2*t["tp"]+t["fp"]+t["fn"],1),4),
                fp_per_case=round(t["fp"]/n,3),
                f1_ci95=bootstrap_ci(rows, n_draws=2000))

for mode in ("default", "nested"):
    t2 = json.load(open(f"{_P}/extern/auswertung_shiva_t2star.json"))["per_case"][mode]
    sw = json.load(open(f"{_P}/extern/auswertung_shiva_swi.json"))["per_case"][mode]
    allr = t2 + sw
    print(mode, "all:", pool(allr))
    print(mode, "t2star:", json.load(open(f"{_P}/extern/auswertung_shiva_t2star.json"))[f"threshold_{mode}"][ "f1"],
          "swi:", json.load(open(f"{_P}/extern/auswertung_shiva_swi.json"))[f"threshold_{mode}"]["f1"])
