#!/bin/bash
# Friday queue (user request 2026-10-07 18:00): the remaining leave-one-cohort-out measurements, started by the systemd user timer
# cmb-freitag.timer on 2026-10-09 06:00 without a session. Every GPU step goes through grossauftrag --vram 20 (stops the language-model
# server, runs, restores it). Everything is resumable: finished trainings (meta.json ok) and predictions are skipped on a rerun.
#
#   A  leave-one-cohort-out over the four T2* cohorts, SECOND seed (1), then the two-seed ensemble scored  (4 trainings)
#   B  leave-one-cohort-out over all SIX cohorts, T2* and SWI together (VALDO 1/2/3, in-house T2*, in-house SWI, Momeni SWI), seed 42  (6 trainings)
#   C  SWI pair: Momeni-only model on in-house SWI, in-house-SWI model on Momeni (full-data models, seed 42)  (2 trainings)
#   D  nnU-Net v2 baseline (arm 10): score the out-of-fold predictions  (CPU)
# Results: ergebnisse/freitag/<job>/..., tables in ergebnisse/freitag/SUMMARY.txt, log ergebnisse/freitag/log.txt; Telegram message at the end.
set -u
R=/home/uchralt/data/work/valdo-t2s; MB=/home/uchralt/data/work/mb-arena; MOM=/home/uchralt/data/extern/momeni-csiro-cmb
P=/home/uchralt/miniconda3/envs/dl/bin/python; G=/home/uchralt/local_agentic_system/system/werkzeuge/grossauftrag; W=/home/uchralt/local_agentic_system/system/werkzeuge
F=$R/ergebnisse/freitag; mkdir -p $F/manifests; cd $R
exec >> $F/log.txt 2>&1
log(){ echo "### $(date '+%F %T') $*"; }
SUM=$F/SUMMARY.txt; : > $SUM
log "START queue"

# ---- manifests: six cohorts, one file each; training = union of the others
$P - <<'PY'
import json, os
MB='/home/uchralt/data/work/mb-arena'; MOM='/home/uchralt/data/extern/momeni-csiro-cmb'; F='/home/uchralt/data/work/valdo-t2s/ergebnisse/freitag/manifests'
v=json.load(open('dev/manifest_valdo_all72.json'))
t2=json.load(open(f'{MB}/gitter/manifest_t2star.json')); sw=json.load(open(f'{MB}/gitter/manifest_swi.json')); mo=json.load(open(f'{MOM}/gitter/manifest_momeni.json'))
coh={'valdo-1':[e for e in v if e['cohort']=='valdo-1'],'valdo-2':[e for e in v if e['cohort']=='valdo-2'],'valdo-3':[e for e in v if e['cohort']=='valdo-3'],
     'charite-t2':t2,'charite-swi':sw,'momeni-swi':mo}
assert len({e['id'] for c in coh.values() for e in c})==sum(len(c) for c in coh.values()), 'ids not unique'
for k,c in coh.items():
    json.dump(c, open(f'{F}/test_{k}.json','w'))
    json.dump([e for kk,cc in coh.items() if kk!=k for e in cc], open(f'{F}/train_six_without_{k}.json','w'))
t2k=['valdo-1','valdo-2','valdo-3','charite-t2']
for k in t2k:
    json.dump([e for kk in t2k if kk!=k for e in coh[kk]], open(f'{F}/train_t2_without_{k}.json','w'))
print({k:(len(c), sum(e.get('n_cmb',0) for e in c)) for k,c in coh.items()})
PY

declare -A SYN=([valdo-1]="dev/synthseg/{id}_synthseg.nii.gz" [valdo-2]="dev/synthseg/{id}_synthseg.nii.gz" [valdo-3]="dev/synthseg/{id}_synthseg.nii.gz"
                [charite-t2]="$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" [charite-swi]="$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" [momeni-swi]="$MOM/synthseg/{id}_synthseg.nii.gz")

train(){ # out train_manifest seed
  if [ -f "$1/meta.json" ] && grep -q '"ok": true' "$1/meta.json"; then echo "trained: $1"; return 0; fi
  $G --vram 20 -- $P -m cmb.training.full_data train --manifest "$2" --out "$1" --seed "$3"; echo "rc=$?"; }
predict(){ # out test_manifest models...
  out=$1; man=$2; shift 2
  $G --vram 20 -- $P -m cmb.training.full_data predict --models "$@" --manifest "$man" --out "$out"; echo "rc=$?"; }
score(){ # label pred_dir test_manifest cohort
  local name=$1 pred=$2 man=$3 k=$4
  echo "== $name" >> $SUM
  $P -m cmb.transfer.evaluate --run $name=$pred --manifest $man 2>&1 | tail -1 | sed 's/by count.*//' | tee -a $SUM
  $P -m cmb.analysis.operating_point --run $name=$pred --manifest $man --synthseg "${SYN[$k]}" --workers 6 2>&1 | grep -E 'fixed cell|CROSS-SELECTED' | tee -a $SUM
  [ -d $pred/stage2 ] || $P -m cmb.stage2.candidates --predictions $pred --manifest $man --synthseg none --keep-csf --out $pred/stage2 > $pred/stage2.log 2>&1
  $P - "$pred/stage2" <<'PY' | tee -a $SUM
import json, numpy as np, torch, sys, torch.nn as nn
sys.path.insert(0,'/home/uchralt/data/work/valdo-t2s')
from cmb.stage2.apply import load, scores_of
from cmb.stage2.final_classifier import build_net
from cmb.stage2.classifier import CROP
b=torch.load('/home/uchralt/data/work/valdo-t2s/ergebnisse/stage2/final_classifier/classifier.pt', map_location='cpu', weights_only=False)
nets=[]
for sd in b['members']:
    n=build_net(nn); n.load_state_dict(sd); nets.append(n.eval())
info, patches, rows, y, fold, n_ref, n_cases = load(sys.argv[1])
z0,y0,x0=((s-c)//2 for s,c in zip(patches.shape[-3:],CROP))
x=torch.from_numpy(np.ascontiguousarray(patches[..., z0:z0+CROP[0], y0:y0+CROP[1], x0:x0+CROP[2]], dtype=np.float32))
with torch.no_grad():
    sc=np.mean([torch.cat([torch.sigmoid(n(x[i:i+128]).squeeze(1)) for i in range(0,len(x),128)]).numpy() for n in nets],axis=0)
r3=scores_of(rows, sc>=0.3, n_ref, n_cases); r5=scores_of(rows, sc>=0.5, n_ref, n_cases)
print(f"stage2 (released classifier): cand {len(rows)}  F1@0.30 {r3['f1']:.3f} (P {r3['precision']:.2f} S {r3['sensitivity']:.2f} FP/case {r3['fp_per_case']:.2f})  F1@0.50 {r5['f1']:.3f}")
json.dump(dict(candidates=len(rows), at_0_3=r3, at_0_5=r5), open(sys.argv[1]+'/stage2_scores.json','w'), indent=1)
PY
}

# ---- A: T2* leave-one-cohort-out, second seed and ensemble
log "A: T2* leave-one-cohort-out, seed 1 + ensemble"
for k in valdo-1 valdo-2 valdo-3 charite-t2; do
  train $F/A/without_$k/seed1 $F/manifests/train_t2_without_$k.json 1
  S42=$R/ergebnisse/loco_t2/without_$k/seed42/modell.pt
  [ -f $S42 ] || { train $F/A/without_$k/seed42 $F/manifests/train_t2_without_$k.json 42; S42=$F/A/without_$k/seed42/modell.pt; }
  predict $F/A/pred_seed1_$k $F/manifests/test_$k.json $F/A/without_$k/seed1/modell.pt
  predict $F/A/pred_ens_$k $F/manifests/test_$k.json $S42 $F/A/without_$k/seed1/modell.pt
  score A-seed1-$k $F/A/pred_seed1_$k $F/manifests/test_$k.json $k
  score A-ensemble-$k $F/A/pred_ens_$k $F/manifests/test_$k.json $k
done

# ---- B: six cohorts, T2* and SWI together
log "B: six-cohort leave-one-out (T2* + SWI)"
for k in valdo-1 valdo-2 valdo-3 charite-t2 charite-swi momeni-swi; do
  train $F/B/without_$k/seed42 $F/manifests/train_six_without_$k.json 42
  predict $F/B/pred_$k $F/manifests/test_$k.json $F/B/without_$k/seed42/modell.pt
  score B-six-$k $F/B/pred_$k $F/manifests/test_$k.json $k
done

# ---- C: SWI pair
log "C: SWI pair (Momeni <-> in-house SWI), full-data models"
train $F/C/momeni_only/seed42 $F/manifests/test_momeni-swi.json 42
train $F/C/charite_swi_only/seed42 $F/manifests/test_charite-swi.json 42
predict $F/C/pred_momeni-on-charite_swi $F/manifests/test_charite-swi.json $F/C/momeni_only/seed42/modell.pt
predict $F/C/pred_charite_swi-on-momeni $F/manifests/test_momeni-swi.json $F/C/charite_swi_only/seed42/modell.pt
score C-momeni-on-charite_swi $F/C/pred_momeni-on-charite_swi $F/manifests/test_charite-swi.json charite-swi
score C-charite_swi-on-momeni $F/C/pred_charite_swi-on-momeni $F/manifests/test_momeni-swi.json momeni-swi

# ---- D: nnU-Net v2 baseline score (arm 10)
log "D: nnU-Net v2 score"
echo "== D nnU-Net v2" >> $SUM
$P -m cmb.baselines.nnunet_v2 score 2>&1 | tail -5 | tee -a $SUM

log "END queue"
[ -x $W/melden ] && $W/melden "Freitag-Warteschlange fertig ($(date '+%H:%M')): A T2*-LOCO Seed 1 + Ensemble, B Sechs-Kohorten-LOCO, C SWI-Paar, D nnU-Net v2. Tabelle: ergebnisse/freitag/SUMMARY.txt" 2>/dev/null || true
