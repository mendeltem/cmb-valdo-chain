#!/usr/bin/env python3
"""wc5: Regel werten (fixiert 2026-10-25) + visueller Check (3 Faelle/Kohorte, quantitatv)
+ bei "ja": GPU-Schritt 5 + tset_summary EINREIHEN (nicht starten).
Regel: weiter NUR wenn der beste der drei AUCs in mindestens einer Kohorte >= 0.60
UND keine Kohorte alle drei unter 0.45 hat.
Schreibt ergebnisse/stage2/wc5_rest/{entscheid.md, vischeck.txt, einreihen.log}.
"""
from __future__ import annotations
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import os
import re
import subprocess
import sys

V = _R
MB = _P
RES = f"{V}/ergebnisse/stage2/wc5_rest"
def _queue(args):
    """GPU queue of the original machine -> optional CMB_QUEUE command; without it the command after "--" runs directly."""
    q = _os.environ.get("CMB_QUEUE", "")
    return q.split() + args if q else args[args.index("--") + 1:]
PY = _PY
FIELDS = ("atlas_vein_p", "atlas_vein_max3mm", "atlas_vein_dist_mm")

def main() -> None:
    lines = open(f"{RES}/auc_ergebnis.txt").read().splitlines()
    cohorts = {}
    for ln in lines:
        m = re.match(r"^(\w+): \d+ candidates \(\d+ false\); AUC for 'false candidate': (.+)$", ln)
        if m:
            name, rest = m.group(1), m.group(2)
            vals = {}
            for part in rest.split(", "):
                f, a = part.rsplit(" ", 1)
                vals[f] = float(a)
            cohorts[name] = vals

    if len(cohorts) != 3:
        raise SystemExit(f"erwarte 3 Kohorten in auc_ergebnis.txt, habe {sorted(cohorts)}")

    # Regel
    best_anywhere = max(max(v.values()) for v in cohorts.values())
    all_below = [k for k, v in cohorts.items() if max(v.values()) < 0.45]
    rule_ok = best_anywhere >= 0.60 and not all_below
    decision = "JA: weiter zu Schritt 5 (GPU)" if rule_ok else "NEIN: kein GPU-Lauf"

    # AUC-Tabelle
    table = ["| Kohorte | atlas_vein_p | atlas_vein_max3mm | atlas_vein_dist_mm |",
             "|---|---|---|---|"]
    for k in ("valdo", "t2star", "swi"):
        v = cohorts[k]
        table.append(f"| {k} | {v['atlas_vein_p']:.3f} | {v['atlas_vein_max3mm']:.3f} | {v['atlas_vein_dist_mm']:.3f} |")
    tbl = "\n".join(table)

    # Visueller Check (3 Faelle/Kohorte, quantitatv, keine Fall-IDs im Bericht)
    vis = []
    import nibabel as nib
    import numpy as np
    from scipy import ndimage
    plan = [
        ("valdo", f"{V}/dev/atlas_vein", f"{V}/dev/gitter_d", f"{V}/dev/gitter_d/{{c}}/{{c}}_image.nii.gz"),
        ("t2star", f"{MB}/atlas_vein/t2star", f"{MB}/gitter/t2star", f"{MB}/gitter/t2star/{{c}}/{{c}}_image.nii.gz"),
        ("swi", f"{MB}/atlas_vein/swi", f"{MB}/gitter/swi", f"{MB}/gitter/swi/{{c}}/{{c}}_image.nii.gz"),
    ]
    for name, maps_dir, grid_dir, img_tpl in plan:
        maps = sorted(f for f in os.listdir(maps_dir) if f.endswith("_vein.nii.gz") and not f.startswith("atlas_on_template"))
        picks = [maps[0], maps[len(maps) // 2], maps[-1]]
        rows = []
        for m in picks:
            c = m.replace("_vein.nii.gz", "")
            v = np.asarray(nib.load(f"{maps_dir}/{m}").dataobj, float)
            brain = np.asarray(nib.load(f"{grid_dir}/{c}/{c}_frst.nii.gz").dataobj, float) > 0 \
                if os.path.exists(f"{grid_dir}/{c}/{c}_frst.nii.gz") else (np.asarray(nib.load(img_tpl.format(c=c)).dataobj, float) > 0)
            vein = v >= 0.5
            d = ndimage.distance_transform_edt(~brain, sampling=(0.5, 0.5, 1.0))
            aff = nib.load(f"{maps_dir}/{m}").affine
            ix, iy, _ = np.nonzero(vein)
            mmx = aff[0, 3] + aff[0, 0] * ix
            mmy = aff[1, 3] + aff[1, 1] * iy
            rows.append((float((vein & brain).mean()), float(np.median(d[vein])),
                         float(((np.abs(mmx) <= 3) & (np.abs(mmy) <= 3)).mean())))
        inb = np.median([r[0] for r in rows])
        edge = np.median([r[1] for r in rows])
        mid = np.median([r[2] for r in rows])
        vis.append((name, inb, edge, mid))
    vis_txt = "visueller Check (3 Faelle je Kohorte, quantitatv, keine Fall-IDs):\n"
    for name, inb, edge, mid in vis:
        ok = "OK" if (inb > 0.9 and edge < 5.0) else "PRUEFEN"
        vis_txt += f"- {name}: Venenmaske zu {inb * 100:.1f}% im Hirn (Hirnrand-Distanz Median {edge:.1f} mm), Mittellinienanteil {mid * 100:.2f}%; {ok}\n"

    # Einreihen bei "ja"
    ein = ""
    if rule_ok:
        cmd = _queue([
            "--einreihen", "--vram", "12", "--", "bash", "-c",
            f"cd {V} && {PY} -m cmb.stage2.pooled --cohort "
            f"valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-atlas "
            f"t2star={MB}/stage2/a03-aniso-e60-cat-t2s-weich-z40-atlas "
            f"swi={MB}/stage2/a03-aniso-e60-cat-swi-weich-z40-atlas "
            f"--folds swi={MB}/gitter/manifest_swi_mix.json --models tset-atlas --skip-own "
            f"--tissue --seed-offset 0 --json {MB}/stage2/tset_tset-atlas_s0.json "
            f"> {MB}/stage2/tset_tset-atlas_s0.log 2>&1",
        ])
        r = subprocess.run(cmd, capture_output=True, text=True)
        cmd2 = _queue([
            "--einreihen", "--", "bash", "-c",
            f"cd {V} && {PY} -m cmb.stage2.tset_summary --json {MB}/stage2/tset_tset-atlas_s0.json "
            f"{MB}/stage2/tset_tset-ctx_s0.json --control 'pooled tset-ctx' "
            f"--out {MB}/stage2/tset_summary_atlas.json",
        ])
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        ein = (f"Eingereiht (rc {r.returncode}/{r2.returncode}):\n"
               f"1. pooled tset-atlas (GPU, --vram 12) -> {MB}/stage2/tset_tset-atlas_s0.json\n"
               f"2. tset_summary --control 'pooled tset-ctx' ueber tset-atlas + tset-ctx "
               f"(wc4) -> {MB}/stage2/tset_summary_atlas.json\n"
               f"Warteschlange bleibt hinter einem anderen Projekt und wc4; Seed 1 nur bei Entscheidung 'advance to seed 1'.\n")
        if r.returncode or r2.returncode:
            ein += f"EINREIHEN-PROBLEM: pooled {r.stdout} {r.stderr} | summary {r2.stdout} {r2.stderr}\n"

    md = f"""# wc5: Regel-Ergebnis + visueller Check + (ggf.) Einreihen

**AUC-Tabelle (AUC fuer 'falscher Kandidat'; hoeher = Vene trennt falsch von echt):**

{tbl}

**Regel (fixiert 2026-10-25):** beste der drei AUCs in mindestens einer Kohorte >= 0.60
({best_anywhere:.3f}) UND keine Kohorte alle drei < 0.45 (alle-drei-unter-0.45: {all_below or 'keine'}).
**Ergebnis: {decision}.**

{vis_txt}
{ein}
"""
    with open(f"{RES}/entscheid.md", "w") as f:
        f.write(md)
    with open(f"{RES}/vischeck.txt", "w") as f:
        f.write(vis_txt)
    print(md)

if __name__ == "__main__":
    sys.exit(main())
