#!/bin/bash
# wc3 Phase 0 (CPU) - Teil A: Unit-Tests (erledigt) + B: Ensembles x3/x4/x4b, Ceiling, Kandidaten, gepoolt, paarweise
# Jeder Schritt wird mit Zeitstempeln geloggt; rc wird gespeichert, die Kette laeuft weiter.
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
PY=${CMB_PYTHON:-python}
V=$CMB_ROOT
MB=$CMB_PRIVATE
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$V"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "=== B1: Ensembles ==="
for spec in \
  "wc3-ensx3:ergebnisse/turnier/a03-aniso-e60-gd:ergebnisse/turnier/a03-aniso-s1-e60-gd:ergebnisse/turnier/a11-anisoatt-e60-gd" \
  "wc3-ensx4:ergebnisse/turnier/a03-aniso-e60-gd:ergebnisse/turnier/a03-aniso-s1-e60-gd:ergebnisse/turnier/a11-anisoatt-e60-gd:ergebnisse/turnier/a15-nnunet-resenc-e60-gd" \
  "wc3-ensx4b:ergebnisse/turnier/a03-aniso-e60-gd:ergebnisse/turnier/a03-aniso-s1-e60-gd:ergebnisse/turnier/a14-mednext-e60-gd:ergebnisse/turnier/a15-nnunet-resenc-e60-gd"; do
  out="${spec%%:*}"; rest="${spec#*:}"
  runs=""; IFS=':'; set -- $rest; for r in "$@"; do runs="$runs $r"; done; unset IFS
  log "ensemble $out"
  $PY -m cmb.analysis.ensemble --runs $runs --out "ergebnisse/turnier/$out"
  log "ensemble $out rc=$?"
done

log "=== B2: Ceiling Ensembles (ens2, x3, x4, x4b) ==="
$PY -m cmb.analysis.candidate_ceiling \
  --run ens2=ergebnisse/turnier/ens2-a03-aniso-e60-gd x3=ergebnisse/turnier/wc3-ensx3 \
        x4=ergebnisse/turnier/wc3-ensx4 x4b=ergebnisse/turnier/wc3-ensx4b \
  --grid dev/gitter_d --thresholds 0.15 0.10 \
  --json ergebnisse/analysis/wc3_ceiling_ensembles.json
log "ceiling-ensembles rc=$?"

log "=== B3: Kandidaten (VALDO, weich) fuer ens2 + x3 + x4 + x4b ==="
for E in ens2-a03-aniso-e60-gd wc3-ensx3 wc3-ensx4 wc3-ensx4b; do
  log "candidates wc3-$E-weich"
  $PY -m cmb.stage2.candidates --predictions "ergebnisse/turnier/$E" --grid dev/gitter_d --keep-csf \
    --out "ergebnisse/stage2/wc3-$E-weich"
  log "candidates $E rc=$?"
done

log "=== B4: pooled boosting je Ensemble (VALDO-Kohorte = jeweiliger Ensemble-Kandidatensatz) ==="
for E in ens2-a03-aniso-e60-gd wc3-ensx3 wc3-ensx4 wc3-ensx4b; do
  log "pooled wc3_ens_$E"
  $PY -m cmb.stage2.pooled \
    --cohort valdo="ergebnisse/stage2/wc3-$E-weich" \
            t2star="$MB/stage2/a03-aniso-e60-cat-t2s-weich" \
            swi="$MB/stage2/a03-aniso-e60-cat-swi-weich" \
    --folds swi="$MB/gitter/manifest_swi_mix.json" \
    --models boosting --skip-own --tissue \
    --json "$MB/stage2/wc3_ens_$E.json"
  log "pooled $E rc=$?"
done

log "=== B5: paarweise gegen ens2 ==="
for E in wc3-ensx3 wc3-ensx4 wc3-ensx4b; do
  $PY -m cmb.stage2.compare_runs --a "$MB/stage2/wc3_ens_$E.json" "probability+csf" \
    --b "$MB/stage2/wc3_ens_ens2-a03-aniso-e60-gd.json" "probability+csf" --cohorts valdo
  $PY -m cmb.stage2.compare_runs --a "$MB/stage2/wc3_ens_$E.json" "pooled+tissue boosting" \
    --b "$MB/stage2/wc3_ens_ens2-a03-aniso-e60-gd.json" "pooled+tissue boosting" --cohorts valdo
  log "compare $E rc=$?"
done
log "=== TEIL A fertig ==="
