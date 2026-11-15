#!/usr/bin/env bash
# wk9 STEP 0/1: one pooled VALDO-only run for one seed. All 5 final arms in one file
# (probability + probability+csf are built-ins of cmb.stage2.pooled, --tissue adds csf).
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -uo pipefail
SEED=$1
ROOT=$CMB_ROOT
PY=${CMB_PYTHON:-python}
cd "$ROOT"
CUDA_VISIBLE_DEVICES="" "$PY" -m cmb.stage2.pooled \
  --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 \
  --tissue --skip-own \
  --models tset-plain feat-vessel feat-vessel-tissue \
  --seed-offset "$SEED" \
  --json $CMB_PRIVATE/stage2/a1v2/tset_a1v2_all_s${SEED}.json
echo "seed $SEED rc=$?"
