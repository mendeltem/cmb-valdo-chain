#!/bin/bash
# Smoke run of every command on synthetic phantoms:  bash configs/smoke.sh cpu|cuda [OUT]   (from the repository root)
# cpu: ~15 min on 8 cores; cuda: a few minutes. Tiny settings (1 epoch, 8 patches) -- checks that it runs, not accuracy.
set -euo pipefail
DEV=${1:-cpu}; OUT=${2:-runs/smoke_$DEV}
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=""
python -m cmb.synthetic "$OUT/phantoms" --n 6
cmb preprocess -i "$OUT/phantoms/raw" -ir _T2starw -f True -b "$OUT/phantoms/raw" -br _brainmask -l "$OUT/phantoms/raw" -lr _cmb -o "$OUT/grid" -fold 2
cmb train -i "$OUT/grid" -m "$OUT/model" -ep 1 --patches 8 --seeds 42 --device "$DEV"
cmb evaluate -i "$OUT/grid" -m "$OUT/model" -o "$OUT/pred" --device "$DEV"
cmb fine_tune -i "$OUT/grid" -m "$OUT/model" -o "$OUT/ft" -ep 1 --patches 8 --device "$DEV"
cmb fine_tune -i "$OUT/grid" -m "$OUT/model" -o "$OUT/ftn" --ft_layers norm_out -ep 1 --patches 8 --device "$DEV"
cmb fine_tune -i "$OUT/grid" -m "$OUT/model" -o "$OUT/cal" --calibrate_only True --device "$DEV"
cmb cross_validate -i "$OUT/grid" -o "$OUT/cv" -fold 5 -ep 1 --patches 8 --device "$DEV"   # add --stage2 joint with a real model (a 1-epoch model proposes no candidates)
cmb predict -i "$OUT/phantoms/raw/syn-001_T2starw.nii.gz" -o "$OUT/predict" -m "$OUT/ft" --device "$DEV" --no-bias-correction
echo "SMOKE $DEV OK -> $OUT"
