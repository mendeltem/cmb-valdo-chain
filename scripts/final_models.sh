#!/bin/bash
# Deliverable models for the product package (user request 2026-10-07): the anisotropic U-Net trained on ALL cases
# (no folds, last state of the 60-epoch schedule), two seeds per cohort; cross tests (VALDO model on the in-house
# cohorts, in-house models on VALDO); the final second-stage classifier on the pooled candidates of all cohorts.
# Every GPU step goes through the job queue (grossauftrag --vram 20: stops the language-model server, runs, restores it).
set -u
R=/home/uchralt/data/work/valdo-t2s; MB=/home/uchralt/data/work/mb-arena
P=/home/uchralt/miniconda3/envs/dl/bin/python; G=/home/uchralt/local_agentic_system/system/werkzeuge/grossauftrag
F=$R/ergebnisse/final; mkdir -p $F/cross; cd $R
log(){ echo "### $(date '+%F %T') $*"; }
VALDO=dev/manifest_valdo_all72.json; T2S=$MB/gitter/manifest_t2star.json; SWI=$MB/gitter/manifest_swi.json
SYN_VALDO="dev/synthseg/{id}_synthseg.nii.gz"; SYN_T2S="$MB/synthseg-t2star/robust/{id}_synthseg.nii.gz"; SYN_SWI="$MB/synthseg-swi/robust/{id}_synthseg.nii.gz"

log "final classifier on the pooled out-of-fold candidates of all three cohorts"
$G --vram 20 -- $P -m cmb.stage2.final_classifier --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich \
   t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich swi=$MB/stage2/a03-aniso-e60-cat-swi-weich --out ergebnisse/stage2/final_classifier; echo "rc=$?"

train(){ name=$1; shift; for seed in 42 1; do log "train $name seed $seed"; $G --vram 20 -- $P -m cmb.training.full_data train --manifest "$@" --out $F/$name/seed$seed --seed $seed; echo "rc=$?"; done; }
train valdo_t2 $VALDO
train charite_t2 $T2S
train charite_swi $SWI

pred(){ name=$1; out=$2; shift 2; log "predict $name -> $out"; $G --vram 20 -- $P -m cmb.training.full_data predict --models $F/$name/seed42/modell.pt $F/$name/seed1/modell.pt --manifest "$@" --out $out; echo "rc=$?"; }
pred valdo_t2    $F/cross/valdo_t2-on-charite_t2     $T2S
pred valdo_t2    $F/cross/valdo_t2-on-charite_swi    $SWI
pred charite_t2  $F/cross/charite_t2-on-valdo        $VALDO
pred charite_swi $F/cross/charite_swi-on-valdo       $VALDO
pred valdo_t2    $F/cross/valdo_t2-on-valdo          $VALDO     # in-sample, optimistic by construction (sanity only)
pred charite_t2  $F/cross/charite_t2-on-charite_t2   $T2S       # in-sample
pred charite_swi $F/cross/charite_swi-on-charite_swi $SWI       # in-sample

score(){ name=$1; man=$2; syn=$3; log "score $name"; $P -m cmb.transfer.evaluate --run $name=$F/cross/$name --manifest $man 2>&1 | tail -2;
         $P -m cmb.analysis.operating_point --run $name=$F/cross/$name --manifest $man --synthseg "$syn" --workers 6 2>&1 | grep -E 'fixed cell|CROSS-SELECTED'; }
score valdo_t2-on-charite_t2     $T2S   "$SYN_T2S"
score valdo_t2-on-charite_swi    $SWI   "$SYN_SWI"
score charite_t2-on-valdo        $VALDO "$SYN_VALDO"
score charite_swi-on-valdo       $VALDO "$SYN_VALDO"
score valdo_t2-on-valdo          $VALDO "$SYN_VALDO"
score charite_t2-on-charite_t2   $T2S   "$SYN_T2S"
score charite_swi-on-charite_swi $SWI   "$SYN_SWI"
log "END"
