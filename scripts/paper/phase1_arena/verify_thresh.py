import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import numpy as np, json, sys, time
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # sibling scripts (were in the private extern folder)
sys.path.insert(0, _R)
import nibabel as nib
from auswertung import f1_pool, load_case
from cmb.metrics import hits
from scipy.ndimage import label as ndlabel
C26 = np.ones((3,3,3), bool)

cohort = sys.argv[1]
dirr = sys.argv[2]
cases = json.load(open(_CASES))[cohort]["faelle"]
rows = []
for c in cases:
    p, r = load_case(f"{dirr}/{c['id']}_prob.nii.gz", c["maske"])
    rows.append((int(c["fold"]), p, r))
print("prob value histogram (all cases):")
vals = np.concatenate([p.ravel() for _, p, _ in rows])
for q in (0,1,5,50,95,99,100):
    print(f"  p{q} = {np.percentile(vals,q):.4f}")
print("  fraction >0:", (vals>0).mean(), " fraction >0.01:", (vals>0.01).mean(), " fraction ==1:", (vals==1).mean())

# brute force on ONE fold's training set (the 4 other folds), full grid
import itertools
target_fold = 0
train = [(p, r) for f, p, r in rows if f != target_fold]
best = []
t0=time.time()
for t in np.arange(0.0, 1.001, 0.01):
    rows2 = [hits(r, p > t) for p, r in train]
    best.append((float(t), round(f1_pool(rows2),4)))
print("brute grid (t, f1) for target fold 0 training, top 8:")
for t, f in sorted(best, key=lambda x:-x[1])[:8]:
    print(f"  t={t:.2f} f1={f}")
print("  at t=0.00:", [b for b in best if b[0]<0.001], " at t=0.50:", [b for b in best if abs(b[0]-0.5)<0.001])
print("Dauer", round(time.time()-t0,1),"s")
