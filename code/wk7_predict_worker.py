#!/usr/bin/env python3
"""wk7: EINE Prognose (Modell, Dicke, Fall) auf der CPU -- fuer paralleles Starten.

    WK7_MODELLPFAD=<ordner mit fold0..4/modell.pt> python code/wk7_predict_worker.py \
        <name> <dicke> <manifest> <case-id> <out-dir>

Schreibt <out-dir>/<name>/<dicke>/fold<k>/<id>_pred_proba.nii.gz (uint8, p*255) mit
dem Modell der Testfalte des Falls (out-of-fold, keine Leakage). Wiederaufnehmbar
(Datei vorhanden -> sofort ok). CPU: CV5_DEV=cpu, OMP_NUM_THREADS aus der Umgebung.
"""
import os, sys, json
import numpy as np
ROOT = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/code")
os.environ["CV5_DEV"] = "cpu"   # CPU inference


def main():
    name, dicke, manifest, sid, mpfad, out = sys.argv[1:7]
    man = json.load(open(manifest))
    rec = next(e for e in man if e["id"] == sid)
    k = rec["fold"]
    p = f"{out}/{name}/{dicke}/fold{k}/{sid}_pred_proba.nii.gz"
    if os.path.exists(p):
        print(f"{name}/{dicke}/{sid}: vorhanden, uebersprungen"); return
    import torch
    import cv5
    import turnier
    import nibabel as nib
    from netze import NETZE
    net = NETZE["a03-aniso"](2)
    net.load_state_dict(torch.load(f"{mpfad}/fold{k}/modell.pt", map_location="cpu"))
    net.eval()
    lader = turnier.mit_manifest([manifest])
    lader()
    proba, nka = cv5.proba_fall(net, sid, bs=16, nch=2)
    r = cv5.FA["recs"][sid]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    back = np.ascontiguousarray(proba.transpose(2, 1, 0))   # BUGFIX 07.11. R4: vorher proba (falsche Achsen)
    nib.save(nib.Nifti1Image((back * 255).round().astype("uint8"), r["affine"]), p)
    print(f"{name}/{dicke}/{sid}: kacheln={nka} max={proba.max():.3f}")


if __name__ == "__main__":
    main()
