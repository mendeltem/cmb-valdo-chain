#!/bin/bash
# wk2 CMB Phase II -- POOLING FACTORIAL, CPU stage-2 (tset-plain) chain.
# 2026-10-31. GPU is held by llama-server (~21 GiB) + maskfix queue item [388], so the
# stage-2 plain-CNN arms run on CPU (tset-plain has no residual/track modules -> CPU ok).
# Stage-1 unified arms (P/S, P/P, PT-FT) are GPU-only and stay PENDING (see ARBEIT.md wk2).
# One pooled.py invocation per seed yields ALL CPU arms:
#   own tset-plain            = S/S (separate stage 2)
#   pooled tset-plain         = S/P (mixed stage 2 = control)
#   own/pooled tset-plain -prob      = C-prob mechanism arms
#   own/pooled tset-plain -prob-frst = C-pf mechanism arms
#   matched tset-plain valdo/t2star/swi = M-match size-matched control
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
V=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-atlas
T=$CMB_PRIVATE/stage2/a03-aniso-e60-cat-t2s-weich-z40-atlas
S=$CMB_PRIVATE/stage2/a03-aniso-e60-cat-swi-weich-z40-atlas
O=$CMB_ROOT/ergebnisse/phase2
mkdir -p "$O"
for seed in 0 1 2; do
  echo "==== wk2 seed $seed $(date +%H:%M:%S) ===="
  $PY -m cmb.stage2.pooled --cohort valdo=$V t2star=$T swi=$S \
      --models tset-plain --channels prob,prob-frst \
      --match-size-to valdo,t2star,swi --seed-offset $seed \
      --json "$O/phase2_seed$seed.json" > "$O/phase2_seed$seed.log" 2>&1
  echo "==== wk2 seed $seed rc=$? $(date +%H:%M:%S) ===="
done
# produce the Phase-2 summary files + figures from the per-seed per-case JSONs
$PY scripts/paper/phase1_arena/phase2_summarise.py > "$O/phase2_summarise.log" 2>&1
echo "==== wk2 summarise rc=$? $(date +%H:%M:%S) ===="
echo "==== wk2 CPU factorial DONE $(date +%H:%M:%S) ===="
touch "$O/phase2_chain.done"
