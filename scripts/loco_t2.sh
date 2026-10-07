#!/bin/bash
# Leave-one-cohort-out over the four T2* cohorts (VALDO 1/2/3, in-house T2*): train on three (all cases, seed 42, 60 epochs, last state),
# predict the left-out cohort, score stage 1 (fixed cell, CSF rule, cross-selected) and stage 2 (released classifier). User question 2026-10-07 17:40.
R=/home/uchralt/data/work/valdo-t2s; MB=/home/uchralt/data/work/mb-arena; P=/home/uchralt/miniconda3/envs/dl/bin/python; G=/home/uchralt/local_agentic_system/system/werkzeuge/grossauftrag
F=$R/ergebnisse/loco_t2; mkdir -p $F; cd $R
log(){ echo "### $(date '+%F %T') $*"; }
declare -A SYN=([valdo-1]="dev/synthseg/{id}_synthseg.nii.gz" [valdo-2]="dev/synthseg/{id}_synthseg.nii.gz" [valdo-3]="dev/synthseg/{id}_synthseg.nii.gz" [charite-t2]="$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz")
for k in valdo-1 valdo-2 valdo-3 charite-t2; do
  log "train without $k"; $G --vram 20 -- $P -m cmb.training.full_data train --manifest dev/loco_t2/train_without_$k.json --out $F/without_$k/seed42 --seed 42; echo "rc=$?"
  log "predict $k"; $G --vram 20 -- $P -m cmb.training.full_data predict --models $F/without_$k/seed42/modell.pt --manifest dev/loco_t2/test_$k.json --out $F/pred_$k; echo "rc=$?"
  log "score $k"
  $P -m cmb.transfer.evaluate --run $k=$F/pred_$k --manifest dev/loco_t2/test_$k.json 2>&1 | tail -1
  $P -m cmb.analysis.operating_point --run $k=$F/pred_$k --manifest dev/loco_t2/test_$k.json --synthseg "${SYN[$k]}" --workers 6 2>&1 | grep -E 'fixed cell|CROSS-SELECTED'
  $P -m cmb.stage2.candidates --predictions $F/pred_$k --manifest dev/loco_t2/test_$k.json --synthseg none --keep-csf --out $F/stage2_$k > $F/stage2_$k.log 2>&1; echo "candidates rc=$?"
done
log "stage2"
$P - <<'PY'
import json, numpy as np, torch, sys, torch.nn as nn
sys.path.insert(0,'.')
from cmb.stage2.apply import load, scores_of
from cmb.stage2.final_classifier import build_net
from cmb.stage2.classifier import CROP
b=torch.load('ergebnisse/stage2/final_classifier/classifier.pt', map_location='cpu', weights_only=False)
dev='cuda' if torch.cuda.is_available() else 'cpu'
nets=[]
for sd in b['members']:
    n=build_net(nn); n.load_state_dict(sd); nets.append(n.eval().to(dev))
out={}
for k in ['valdo-1','valdo-2','valdo-3','charite-t2']:
    info, patches, rows, y, fold, n_ref, n_cases = load(f'ergebnisse/loco_t2/stage2_{k}')
    z0,y0,x0=((s-c)//2 for s,c in zip(patches.shape[-3:],CROP))
    x=torch.from_numpy(np.ascontiguousarray(patches[..., z0:z0+CROP[0], y0:y0+CROP[1], x0:x0+CROP[2]], dtype=np.float32))
    with torch.no_grad():
        sc=np.mean([torch.cat([torch.sigmoid(n(x[i:i+256].to(dev)).squeeze(1)).cpu() for i in range(0,len(x),256)]).numpy() for n in nets],axis=0)
    res={str(t):{kk:round(vv,3) for kk,vv in scores_of(rows, sc>=t, n_ref, n_cases).items()} for t in (0.3,0.5)}
    out[k]=dict(candidates=len(rows), reference=n_ref, cases=n_cases, classifier=res)
    print(f"{k:12s} cand {len(rows):5d}  F1@0.30 {res['0.3']['f1']:.3f} (P {res['0.3']['precision']:.2f} S {res['0.3']['sensitivity']:.2f} FP {res['0.3']['fp_per_case']:.1f})  F1@0.50 {res['0.5']['f1']:.3f}")
json.dump(out, open('ergebnisse/loco_t2/stage2_scores.json','w'), indent=1)
PY
log "END"
