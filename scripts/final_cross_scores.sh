#!/bin/bash
# Complete scoring of the cross tests: CSF rule (SynthSeg pattern with {sid} for the in-house cohorts) and the product's stage 2.
R=/home/uchralt/data/work/valdo-t2s; MB=/home/uchralt/data/work/mb-arena; P=/home/uchralt/miniconda3/envs/dl/bin/python; F=$R/ergebnisse/final; cd $R
VALDO=dev/manifest_valdo_all72.json; T2S=$MB/gitter/manifest_t2star.json; SWI=$MB/gitter/manifest_swi.json
declare -A MAN=([valdo_t2-on-charite_t2]=$T2S [valdo_t2-on-charite_swi]=$SWI [charite_t2-on-valdo]=$VALDO [charite_swi-on-valdo]=$VALDO [valdo_t2-on-valdo]=$VALDO [charite_t2-on-charite_t2]=$T2S [charite_swi-on-charite_swi]=$SWI)
declare -A SYN=([valdo_t2-on-charite_t2]="$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" [valdo_t2-on-charite_swi]="$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" [charite_t2-on-valdo]="dev/synthseg/{id}_synthseg.nii.gz" [charite_swi-on-valdo]="dev/synthseg/{id}_synthseg.nii.gz" [valdo_t2-on-valdo]="dev/synthseg/{id}_synthseg.nii.gz" [charite_t2-on-charite_t2]="$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" [charite_swi-on-charite_swi]="$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz")
for n in "${!MAN[@]}"; do
  echo "### score $n"
  $P -m cmb.transfer.evaluate --run $n=$F/cross/$n --manifest ${MAN[$n]} 2>&1 | tail -1
  $P -m cmb.analysis.operating_point --run $n=$F/cross/$n --manifest ${MAN[$n]} --synthseg "${SYN[$n]}" --workers 6 2>&1 | grep -E 'fixed cell|CROSS-SELECTED'
  [ -d $F/stage2/$n ] || $P -m cmb.stage2.candidates --predictions $F/cross/$n --manifest ${MAN[$n]} --synthseg none --keep-csf --out $F/stage2/$n > $F/stage2_$n.log 2>&1
done
echo "### stage2"
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
for name in ['valdo_t2-on-charite_t2','valdo_t2-on-charite_swi','charite_t2-on-valdo','charite_swi-on-valdo','valdo_t2-on-valdo','charite_t2-on-charite_t2','charite_swi-on-charite_swi']:
    info, patches, rows, y, fold, n_ref, n_cases = load(f'ergebnisse/final/stage2/{name}')
    z0,y0,x0=((s-c)//2 for s,c in zip(patches.shape[-3:],CROP))
    x=torch.from_numpy(np.ascontiguousarray(patches[..., z0:z0+CROP[0], y0:y0+CROP[1], x0:x0+CROP[2]], dtype=np.float32))
    with torch.no_grad():
        sc=np.mean([torch.cat([torch.sigmoid(n(x[i:i+256].to(dev)).squeeze(1)).cpu() for i in range(0,len(x),256)]).numpy() for n in nets],axis=0)
    res={}
    for t in (0.3,0.4,0.5,0.6):
        res[str(t)]={k:round(v,3) for k,v in scores_of(rows, sc>=t, n_ref, n_cases).items()}
    out[name]=dict(candidates=len(rows), reference=n_ref, cases=n_cases, classifier=res)
    print(f"{name:28s} cand {len(rows):5d}  F1@0.30 {res['0.3']['f1']:.3f} (P {res['0.3']['precision']:.2f} S {res['0.3']['sensitivity']:.2f} FP {res['0.3']['fp_per_case']:.1f})  F1@0.50 {res['0.5']['f1']:.3f}")
json.dump(out, open('ergebnisse/final/stage2_scores.json','w'), indent=1)
PY
echo "### END"
