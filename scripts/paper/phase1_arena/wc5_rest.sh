#!/bin/bash
# wc5-rest: Schritt 3-5 automatisch, WIEDERAUFNEHMBAR (Gutachter-Nachbesserung R3, 30.10.).
# Waechst auf die fertigen Register-Maps (valdo 57, t2star 75, swi 61), dann je Kohorte
# 'features' (ersetzt die Platzhalter in den -atlas-Ordnern), dann 'auc' (30-min-Check),
# dann die Regel (fixiert 2026-10-25): weiter NUR wenn der beste der drei AUCs in >=1
# Kohorte >= 0.60 UND keine Kohorte alle drei < 0.45. Bei "ja": GPU-Schritt 5 EINREIHEN
# (pooled tset-atlas, seed-offset 0) + tset_summary einreihen (CPU, braucht das
# GPU-Ergebnis-JSON). Alle Zwischenergebnisse nach ergebnisse/stage2/wc5_rest/ .
# momeni wird nur registriert (Kette wc5-register), hier NICHT features/lauf.
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -u
PY=${CMB_PYTHON:-python}
MB=$CMB_PRIVATE
V=$CMB_ROOT
OUT=$V/ergebnisse/stage2
RES=$OUT/wc5_rest
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$V" || exit 1
mkdir -p "$RES"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RES/rest.log"; }

maps(){ ls "$1"/*_vein.nii.gz 2>/dev/null | grep -vc atlas_on_template || true; }

log "wc5-rest startet; Maps: valdo=$(maps dev/atlas_vein) t2star=$(maps $MB/atlas_vein/t2star) swi=$(maps $MB/atlas_vein/swi)"

# 1) Warten, bis alle 3 Kohorten registriert sind (Register-Laeufe sind wiederaufnehmbar).
while :; do
  nv=$(maps dev/atlas_vein); nt=$(maps $MB/atlas_vein/t2star); ns=$(maps $MB/atlas_vein/swi)
  if [ "$nv" -ge 57 ] && [ "$nt" -ge 75 ] && [ "$ns" -ge 61 ]; then
    log "Register komplett: valdo=$nv t2star=$nt swi=$ns"
    break
  fi
  # Register-Laeufe tot und nicht fertig? -> neu starten (wiederaufnehmbar).
  if ! pgrep -f "atlas_features register" >/dev/null; then
    log "WARNUNG: kein register-Prozess mehr laeuft (valdo=$nv t2star=$nt swi=$ns); Kette neu gestartet"
    nice -n 10 bash "$CMB_ROOT/scripts/paper/phase1_arena/wc5_register_all.sh" >> "$RES/register.log" 2>&1 &
    disown
    sleep 120
  fi
  sleep 300
done

# 2) features je Kohorte (ersetzt die Platzhalter in den -atlas-Ordnern)
nice -n 10 $PY -m cmb.stage2.atlas_features features \
  --candidates "$OUT/a03-aniso-e60-gd-weich-z40" --maps "dev/atlas_vein/{id}_vein.nii.gz" \
  --out "$OUT/a03-aniso-e60-gd-weich-z40-atlas" >> "$RES/rest.log" 2>&1 \
  || { log "FAIL features valdo"; exit 1; }
nice -n 10 $PY -m cmb.stage2.atlas_features features \
  --candidates "$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40" --maps "$MB/atlas_vein/t2star/{id}_vein.nii.gz" \
  --out "$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40-atlas" >> "$RES/rest.log" 2>&1 \
  || { log "FAIL features t2star"; exit 1; }
nice -n 10 $PY -m cmb.stage2.atlas_features features \
  --candidates "$MB/stage2/a03-aniso-e60-cat-swi-weich-z40" --maps "$MB/atlas_vein/swi/{id}_vein.nii.gz" \
  --out "$MB/stage2/a03-aniso-e60-cat-swi-weich-z40-atlas" >> "$RES/rest.log" 2>&1 \
  || { log "FAIL features swi"; exit 1; }
log "features fertig (Platzhalter durch Echte Werte ersetzt)"

# 3) 30-min-Check: auc (Schritt 4)
nice -n 10 $PY -m cmb.stage2.atlas_features auc \
  --candidates valdo="$OUT/a03-aniso-e60-gd-weich-z40-atlas" \
    t2star="$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40-atlas" \
    swi="$MB/stage2/a03-aniso-e60-cat-swi-weich-z40-atlas" \
  > "$RES/auc_ergebnis.txt" 2>&1 || { log "FAIL auc"; exit 1; }
log "auc fertig -> $RES/auc_ergebnis.txt"

# 4) Regel werten + visueller Check (3 Faelle/Kohorte, quantitatv: Hirnanteil,
#    Hirnrand-Distanz, Mittellinie) + Einreihen bei "ja"
nice -n 10 $PY "$CMB_ROOT/scripts/paper/phase1_arena/wc5_regel_entscheid.py" >> "$RES/rest.log" 2>&1 \
  || { log "FAIL regel/entscheid"; exit 1; }
log "wc5-rest FERTIG (Entscheidung in $RES/entscheid.md, AUC in $RES/auc_ergebnis.txt)"
