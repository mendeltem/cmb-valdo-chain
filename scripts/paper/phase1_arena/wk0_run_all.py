# -*- coding: utf-8 -*-
"""wk0 driver: run all four datasets + merge (CPU, background).

F1@0.3 check numbers come from the transfer-folder auswertung*.json files
(threshold 0.3, >= 2 mm3, giants removed) and fp_location_valdo*.json
(VALDO baselines). Mismatch > 0.005 aborts the run.
"""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import sys, time
sys.path.insert(0, _R)
from cmb.analysis.unet_rerank import run_dataset, merge_reports

ROOT = _R
O = _P + "/paper/unet_rerank"

t_all = time.time()
t0 = time.time()
run_dataset("valdo", f"{ROOT}/ergebnisse/turnier",
            f"{ROOT}/dev/manifest_valdo_gitter_d.json",
            "a03-aniso-e60-gd", O, workers=20,
            expected_f1={"a03-aniso-e60-gd": 0.585,
                         "a03-aniso-e60-gd-vblob": 0.541,
                         "a03-aniso-e60-gd-gew": 0.527})
print(f"valdo done in {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
run_dataset("cat-t2s", _P + "/transfer",
            _P + "/gitter/manifest_t2star.json",
            "a03-aniso-e60-cat-t2s", O, workers=20,
            expected_f1={"a03-aniso-e60-cat-t2s": 0.663,
                         "a03-aniso-e60-vblob-cat-t2s": 0.634,
                         "zero-shot-t2star": 0.622,
                         "zero-shot-t2star-s1": 0.638,
                         "zero-shot-t2star-nlm": 0.574,
                         "zero-shot-t2star-nz": 0.618})
print(f"cat-t2s done in {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
run_dataset("cat-swi", _P + "/transfer",
            _P + "/gitter/manifest_swi.json",
            "a03-aniso-e60-cat-swi", O, workers=20,
            expected_f1={"a03-aniso-e60-cat-swi": 0.634,
                         "a03-aniso-e60-vblob-cat-swi": 0.631,
                         "zero-shot-swi": 0.305,
                         "a03-aniso-s1-e60-cat-swi": 0.613,
                         "zero-shot-swi-nlm": 0.271,
                         "zero-shot-swi-nz": 0.283})
print(f"cat-swi done in {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
run_dataset("cat-swi-mix", _P + "/transfer",
            _P + "/gitter/manifest_swi_mix.json",
            "a03-aniso-e30-lr0.0001-ft-von-mixalle-swi", O, workers=4)
print(f"cat-swi-mix done in {time.time()-t0:.0f}s", flush=True)

merge_reports(O, ["valdo", "cat-t2s", "cat-swi", "cat-swi-mix"])
print(f"ALL DONE in {time.time()-t_all:.0f}s", flush=True)
