"""Acceptance check wk3 (31.10.2026): the three phase-3 summaries must come from the specified classifier tset-plain."""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json, sys
P = _P + "/paper"
bad = [f for f in ("phase3a_summary", "phase3c_summary", "phase3c_fixed_summary")
       if json.load(open(f"{P}/{f}.json")).get("meta", {}).get("model") != "tset-plain"]
print("not tset-plain:", bad if bad else "none"); sys.exit(1 if bad else 0)
