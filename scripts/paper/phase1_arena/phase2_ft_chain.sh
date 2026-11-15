#!/bin/bash
# wk2b CMB Phase II -- P->FT (pooled + fine-tune) chain.
# 2026-11-05. GPU-only (trackset training).
# Same cohorts/folds as wk2 phase2_chain.sh (phase2_seed{0,1,2}.json).
# One pooled.py invocation per seed yields the P->FT arm:
#   pooled-ft tset-plain = pre-train on pooled (all cohorts fold!=k) + fine-tune on D's fold!=k
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -u
PY=${CMB_PYTHON:-python}
cd $CMB_ROOT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
V=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-atlas
T=$CMB_PRIVATE/stage2/a03-aniso-e60-cat-t2s-weich-z40-atlas
S=$CMB_PRIVATE/stage2/a03-aniso-e60-cat-swi-weich-z40-atlas
O=$CMB_ROOT/ergebnisse/phase2
mkdir -p "$O"
for seed in 0 1 2; do
  echo "==== wk2b P->FT seed $seed $(date +%H:%M:%S) ===="
  $PY -m cmb.stage2.pooled --cohort valdo=$V t2star=$T swi=$S \
      --models tset-plain --pooled-finetune \
      --seed-offset $seed \
      --json "$O/phase2_ft_seed$seed.json" > "$O/phase2_ft_seed$seed.log" 2>&1
  echo "==== wk2b P->FT seed $seed rc=$? $(date +%H:%M:%S) ===="
done
# summary + figure from the per-seed per-case JSONs
$PY scripts/paper/phase1_arena/phase2_ft_summarise.py > "$O/phase2_ft_summarise.log" 2>&1
echo "==== wk2b P->FT summarise rc=$? $(date +%H:%M:%S) ===="
echo "==== wk2b P->FT chain DONE $(date +%H:%M:%S) ===="
touch "$O/phase2_ft_chain.done"
