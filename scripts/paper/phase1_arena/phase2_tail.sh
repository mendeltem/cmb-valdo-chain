#!/bin/bash
# wk2 CMB Phase II -- tail: wait for the seed1/seed2 chain, then run the KILLED
# seed0 (rc 143 at 03:52, see phase2_seed0.log), then summarise into the paper
# deliverables (phase2_summary.json, phase2_channels_summary.json,
# phase2_size_summary.json + fig2/fig2b/fig2c). 31.10.2026.
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
for i in $(seq 1 300); do
  if [ -f "$CHAIN" ]; then break; fi
  sleep 60
done
if [ ! -f "$CHAIN" ]; then
  echo "chain marker missing after 300 min -- summarising what exists"
fi
# a seed json is REAL iff the chain marker exists, it is newer than the marker,
# and it contains the matched-mode per-case entries (synthetic test files lack them
# and predate the marker)
real_seed() {
  local s=$1
  [ -f "$O/phase2_seed$s.json" ] && [ -f "$CHAIN" ] \
    && [ "$O/phase2_seed$s.json" -nt "$CHAIN" ] \
    && grep -q '"matched tset-plain' "$O/phase2_seed$s.json"
}
run_seed() {
  $PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-atlas \
      t2star=$CMB_PRIVATE/stage2/a03-aniso-e60-cat-t2s-weich-z40-atlas \
      swi=$CMB_PRIVATE/stage2/a03-aniso-e60-cat-swi-weich-z40-atlas \
      --models tset-plain --channels prob,prob-frst \
      --match-size-to valdo,t2star,swi --seed-offset $1 \
      --json "$O/phase2_seed$1.json" > "$O/phase2_seed$1.log" 2>&1
  echo "==== wk2 seed$1 rc=$? $(date +%H:%M:%S) ===="
}
if real_seed 0; then echo "==== seed0 real, skip $(date +%H:%M:%S) ===="; else run_seed 0; fi
if real_seed 1; then echo "==== seed1 real, skip $(date +%H:%M:%S) ===="; else run_seed 1; fi
if real_seed 2; then echo "==== seed2 real, skip $(date +%H:%M:%S) ===="; else run_seed 2; fi
$PY scripts/paper/phase1_arena/phase2_summarise.py > "$O/phase2_summarise.log" 2>&1
echo "==== wk2 tail summarise rc=$? $(date +%H:%M:%S) ===="
touch "$O/phase2_tail.done"
