#!/usr/bin/env python3
"""wk11 probe: dump structure of phase4 sources. Read-only, no results touched."""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json, os, re, collections

P4 = _P + "/paper/phase4"
chains = json.load(open(os.path.join(P4, "chains_frozenA1v2.json")))
pe = json.load(open(os.path.join(P4, "primary_endpoint.json")))
gfu = json.load(open(os.path.join(P4, "good_for_us.json")))

print("== chains: type", type(chains).__name__, "keys", len(chains))
keys = list(chains)
print("first 5:", keys[:5])
pat = re.compile(r"^(?P<seq>[^/]+)/(?P<variant>[^_]+)_n(?P<n>\d+)_fold(?P<fold>\d+)_s(?P<seed>\d+)(?:_(?P<tag>.+))?$")
bad = [k for k in keys if not pat.match(k)]
print("keys not matching pattern:", len(bad), bad[:5])
comb = collections.Counter()
for k in keys:
    m = pat.match(k)
    if m:
        comb[(m["seq"], m["variant"], int(m["n"]), m["tag"])] += 1
for c in sorted(comb, key=lambda x: (x[0], x[1], x[2], str(x[3]))):
    print("  cell", c, "blocks", comb[c])

print()
print("== chain block example ==")
k0 = keys[0]
b0 = chains[k0]
print("key:", k0)
print("block keys:", list(b0))
def peek(o, d=0):
    if isinstance(o, dict):
        for kk, vv in list(o.items())[:12]:
            if isinstance(vv, (dict, list)):
                print("  " * (d + 1) + f"{kk}: {type(vv).__name__} len={len(vv)}")
                if d < 2:
                    peek(vv, d + 1)
            else:
                print("  " * (d + 1) + f"{kk}: {vv!r}")
    elif isinstance(o, list):
        print("  " * (d + 1) + f"list len={len(o)} first={o[0]!r}" if o else "  " * (d + 1) + "empty list")
peek(b0)

print()
print("== primary_endpoint keys ==")
print(json.dumps(pe, indent=1)[:4000])
print()
print("== good_for_us keys ==")
print(json.dumps(gfu, indent=1)[:3000])
