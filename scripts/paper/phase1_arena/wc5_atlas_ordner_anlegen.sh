#!/bin/bash
# wc5: -atlas-Ordner anlegen BEVOR das Register fertig ist (Abnahmefordderung 30.10.).
# candidates.json = Zeilen der -z40-Quelle + 3 Atlas-Felder mit PLATZHALTER 0.0
# (Register laeuft; echte Werte ersetzt 'features' in der naechsten Runde).
# patches.npy wird verlinkt, nicht kopiert (wie in atlas_features.py).
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -e
PY=${CMB_PYTHON:-python}
$PY - <<'EOF'
import json, os
pairs = [
    (__import__("os").environ["CMB_ROOT"] + "/ergebnisse/stage2/a03-aniso-e60-gd-weich-z40",
     __import__("os").environ["CMB_ROOT"] + "/ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-atlas"),
    (__import__("os").environ["CMB_PRIVATE"] + "/stage2/a03-aniso-e60-cat-t2s-weich-z40",
     __import__("os").environ["CMB_PRIVATE"] + "/stage2/a03-aniso-e60-cat-t2s-weich-z40-atlas"),
    (__import__("os").environ["CMB_PRIVATE"] + "/stage2/a03-aniso-e60-cat-swi-weich-z40",
     __import__("os").environ["CMB_PRIVATE"] + "/stage2/a03-aniso-e60-cat-swi-weich-z40-atlas"),
]
for src, dst in pairs:
    info = json.load(open(f"{src}/candidates.json"))
    for r in info["candidates"]:
        r["atlas_vein_p"] = 0.0
        r["atlas_vein_max3mm"] = 0.0
        r["atlas_vein_dist_mm"] = 20.0
    info["atlas"] = dict(atlas="PLACEHOLDER", maps="(register laeuft)", level=0.5,
                         note="Platzhalter 30.10.2026: Register laeuft; 'features' ersetzt diese Datei, sobald die Maps fertig sind")
    os.makedirs(dst, exist_ok=True)
    json.dump(info, open(f"{dst}/candidates.json", "w"))
    link = f"{dst}/patches.npy"
    if not os.path.exists(link):
        os.symlink(os.path.realpath(f"{src}/patches.npy"), link)
    print(f"OK {dst}: {len(info['candidates'])} Kandidaten, {len(info['cases'])} Faelle")
EOF
