#!/bin/bash
# wk2 CMB Phase II -- tail v2 (31.10.2026 07:18): ersetzt phase2_tail.sh.
# BUG v1: real_seed() verwarf seed-JSONs, die AELTER als phase2_chain.done sind
# (pooled.py schreibt das JSON am ENDE des Laufs; chain.done kommt danach)
# -> wuerde seed1 (laeuft seit 03:52) und seed2 NOCHMAL ~3.3 h laufen lassen.
# v2: ein seed-JSON ist echt, wenn es (a) nach dem START der Kette (03:52)
# entstanden ist und (b) die matched-Modi enthaelt (synthetische Testdateien
# sind nach synthetik/ verschoben, haben aber die matched-Modi NICHT im
# richtigen Format -- Doppelabsicherung: MTIME-Zaun + Strukturcheck).
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -u
PY=${CMB_PYTHON:-python}
cd $CMB_ROOT
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
O=ergebnisse/phase2
CHAIN=ergebnisse/phase2/phase2_chain.done

# Kette wartet auf chain.done (max 540 min), dann Status
for i in $(seq 1 540); do
  if [ -f "$CHAIN" ]; then break; fi
  sleep 60
done
[ -f "$CHAIN" ] || echo "!! chain marker fehlt nach 9 h -- wasche was geht"

real_seed() {
  local s=$1 f=$O/phase2_seed$1.json
  [ -f "$f" ] || return 1
  # Zaun: nach Ketten-Start (seed1 startete 2026-10-31 03:52:58)
  [ "$f" -nt $CMB_LOGS/wk2-phase2-cpu.pid ] || return 1
  # Struktur: matched-Modi muessen pro Kohorte per_case-Eintraege haben
  $PY - "$f" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
pc = d.get("per_case", {})
ok = all(
    all(len(pc.get(m, {}).get(c, {})) > 0 for c in ("valdo", "t2star", "swi"))
    for m in ("own tset-plain", "pooled tset-plain", "matched tset-plain swi")
)
sys.exit(0 if ok else 1)
EOF
}

run_seed() {
  $PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-atlas \
      t2star=$CMB_PRIVATE/stage2/a03-aniso-e60-cat-t2s-weich-z40-atlas \
      swi=$CMB_PRIVATE/stage2/a03-aniso-e60-cat-swi-weich-z40-atlas \
      --models tset-plain --channels prob,prob-frst \
      --match-size-to valdo,t2star,swi --seed-offset $1 \
      --json "$O/phase2_seed$1.json" >> "$O/phase2_seed$1.log" 2>&1
  echo "==== wk2 tail2 seed$1 rc=$? $(date +%H:%M:%S) ===="
}

# chain.done existiert -> chain hat seed1 + seed2 geschrieben (rc 143 seed0 nur)
# chain.done Fehlt (Kette gestorben) -> alles nachrechnen
if [ -f "$CHAIN" ]; then
  for s in 0 1 2; do real_seed $s && echo "==== seed$s real, skip $(date +%H:%M:%S) ====" || run_seed $s; done
else
  echo "==== chain.gestorben: alle seeds neu $(date +%H:%M:%S) ===="
  for s in 0 1 2; do run_seed $s; done
fi

$PY scripts/paper/phase1_arena/phase2_summarise.py >> "$O/phase2_summarise.log" 2>&1
echo "==== wk2 tail2 summarise rc=$? $(date +%H:%M:%S) ===="
touch "$O/phase2_tail2.done"
