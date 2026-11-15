"""wk5 step-1 check: metrics recomputed from the decision dumps equal the stored JSON results (tolerance 1e-6).

Dump: <private>/paper/decisions/chainA_s{0,1}.jsonl (pooled.py --dump-decisions).
Stored: <private>/maskfix/tset_v2_tset-plain_s{0,1}.json (results[mode][cohort]).
Reference counts per case from the candidates files used by the run (gpu_chain_swi_v2_rest.sh step 5):
  valdo  <repo>/ergebnisse/stage2/a03-aniso-e60-gd-weich-z40/candidates.json
  t2star <private>/stage2/a03-aniso-e60-cat-t2s-weich-z40/candidates.json
  swi    <private>/stage2/a03-aniso-e60-cat-swi-v2-weich-z40/candidates.json
Metric = cmb.stage2.pooled.scores_of semantics (TP = reference lesions touched by an accepted candidate).
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
import json
import sys

import numpy as np

V = _R
MB = _P
CAND = {
    "valdo": f"{V}/ergebnisse/stage2/a03-aniso-e60-gd-weich-z40/candidates.json",
    "t2star": f"{MB}/stage2/a03-aniso-e60-cat-t2s-weich-z40/candidates.json",
    "swi": f"{MB}/stage2/a03-aniso-e60-cat-swi-v2-weich-z40/candidates.json",
}
MODES = ("probability", "probability+csf", "pooled tset-plain",
         "pooled tset-plain -prob", "pooled tset-plain -prob-frst")
TOL = 1e-6          # raw tolerance of the spec; the STORED JSON rounds to 3-4 digits, so compare with the rounding slack below
ROUNDING_SLACK = 5e-4


def scores_of(found, fp, n_reference, n_cases):
    tp = len(found)
    fn = n_reference - tp
    return dict(f1=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                sensitivity=tp / max(tp + fn, 1), fp_per_case=fp / max(n_cases, 1))


def main():
    nref, ncases = {}, {}
    for name, path in CAND.items():
        d = json.load(open(path))
        nref[name] = {c: v["n_reference"] for c, v in d["cases"].items()}
        ncases[name] = len(d["cases"])
    for s in (0, 1):
        stored = json.load(open(f"{MB}/maskfix/tset_v2_tset-plain_s{s}.json"))
        rows = [json.loads(l) for l in open(f"{MB}/paper/decisions/chainA_s{s}.jsonl")]
        ok = True
        for mode in MODES:
            sub = [r for r in rows if r["mode"] == mode]
            for name in ("valdo", "t2star", "swi"):
                here = [r for r in sub if r["cohort"] == name]
                # n_reference / n_cases over ALL cohort cases (cases without a candidate still count; the dump
                # holds only candidates, so those cases are absent from it)
                ref_n = sum(nref[name].values())
                found = {(r["case"], l) for r in here if r["accepted"] for l in r["touched"]}
                fp = sum(1 for r in here if r["accepted"] and not r["touched"])
                got = scores_of(found, fp, ref_n, ncases[name])
                want = stored["results"][mode][name]
                for k in ("f1", "precision", "sensitivity", "fp_per_case"):
                    diff = abs(got[k] - want[k])
                    status = "ok" if diff <= TOL else ("rounding" if diff <= ROUNDING_SLACK else "FAIL")
                    if diff > ROUNDING_SLACK:
                        ok = False
                    print(f"s{s} {mode:28s} {name:8s} {k:12s} dump {got[k]:.6f} json {want[k]:.4f} diff {diff:.2e} {status}")
        print(f"chainA_s{s}: {'ALL MODES REPRODUCE THE STORED METRICS (within rounding of the stored 3-4 digit values)' if ok else 'MISMATCH'}",
              flush=True)
        if not ok:
            sys.exit(1)
    print("CHECK1 OK: metrics recomputed from the dump equal the JSON results (stored rounding 1e-3/1e-4, raw tolerance 1e-6)")


if __name__ == "__main__":
    main()
