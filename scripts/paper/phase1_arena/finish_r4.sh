#!/usr/bin/env bash
# wk9 R4: finish the deliverables once the two pooled VALDO-only seeds are done.
# waits for tset_a1v2_all_s0.json and s1.json -> select_a1v2.py (selection.json)
# -> chain_frozenA1v2.py (real chain, winner arm, CPU) -> comparison -> figure.
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -uo pipefail
MB=$CMB_PRIVATE
ROOT=$CMB_ROOT
P4=$MB/paper/phase4
S0=$MB/stage2/a1v2/tset_a1v2_all_s0.json
S1=$MB/stage2/a1v2/tset_a1v2_all_s1.json
PY=${CMB_PYTHON:-python}

echo "$(date) waiting for the two seed runs..."
while [ ! -s "$S0" ] || [ ! -s "$S1" ]; do
  sleep 30
done
echo "$(date) both seed runs done"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="" $PY $CMB_ROOT/scripts/paper/phase1_arena/select_a1v2.py || { echo "select failed"; exit 1; }
echo "$(date) selection.json written"
NCAND=$(ls -d $CMB_PRIVATE/paper/phase4/candidates/*/*/ 2>/dev/null | wc -l)
echo "$(date) candidate block dirs found: $NCAND"
if [ "$NCAND" -ge 1 ]; then
  CUDA_VISIBLE_DEVICES="" $PY $CMB_ROOT/scripts/paper/phase2_finetune/chain_frozenA1v2.py || { echo "chain failed"; exit 1; }
  echo "$(date) chain done"
else
  echo "$(date) SKIP chain_frozenA1v2.py: target candidate sets not locally available (project-mode network guard) -- keeping the 780-entry error file, not an empty {}"
fi
CUDA_VISIBLE_DEVICES="" $PY $CMB_ROOT/scripts/paper/phase2_finetune/compare_a1v2_frozenV.py || true
CUDA_VISIBLE_DEVICES="" $PY $CMB_ROOT/scripts/paper/phase2_finetune/fig_a1v2.py || true
echo "$(date) wk9 R4 chain finished"
