#!/usr/bin/env bash
# wc6 Schritt 6: Realism-Gate (GPU) fuer die drei Kohorten, seed 0.
# Exakt aus dem Auftrag. Ersetzt den Platzhalter $MB/synthetic/realism_s0.json.
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -u
PY=${CMB_PYTHON:-python}
V=$CMB_ROOT
MB=$CMB_PRIVATE

cd "$V"
$PY -m cmb.synthesis.realism_probe \
  --real valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 \
        t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40 \
        swi=$MB/stage2/a03-aniso-e60-cat-swi-weich-z40 \
  --synthetic valdo=ergebnisse/synthetic/valdo-s0/candidates,ergebnisse/synthetic/valdo-s0/insert_log.json \
              t2star=$MB/synthetic/${CMB_LABEL}-t2star-s0/candidates,$MB/synthetic/${CMB_LABEL}-t2star-s0/insert_log.json \
              swi=$MB/synthetic/${CMB_LABEL}-swi-s0/candidates,$MB/synthetic/${CMB_LABEL}-swi-s0/insert_log.json \
  --json $MB/synthetic/realism_s0.json

echo "=== wc6 realism FERTIG rc=$? ==="
