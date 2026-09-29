#!/bin/bash
# SynthSeg --robust for all private SWI cases (Claude 2026-09-21; same settings as the T2* cohort). Pilot on 3 cases: the plain mode puts
# 13 of 54 annotated microbleeds into CSF, the robust mode 3 of 54 -- so robust it is. One case at a time.
export FREESURFER_HOME=/home/uchralt/freesurfer/8.2.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh >/dev/null 2>&1
OUT=/home/uchralt/data/work/mb-arena/synthseg-swi/robust; mkdir -p $OUT
B=/home/uchralt/data/bids/catalina-swi/derivatives/biascorr
for D in $B/sub-*; do
  S=$(basename $D); [ -f $OUT/${S}_synthseg.nii.gz ] && continue
  IMG=$(ls $D/anat/*biascorr*.nii.gz 2>/dev/null | head -1); [ -z "$IMG" ] && { echo "$S: no image" >> $OUT/log.txt; continue; }
  nice -n 10 mri_synthseg --i "$IMG" --o $OUT/${S}_synthseg.nii.gz --vol $OUT/${S}_vol.csv --cpu --threads 6 --robust >> $OUT/log.txt 2>&1 || echo "$S: FAILED rc=$?" >> $OUT/log.txt
done
echo "done: $(ls $OUT/*_synthseg.nii.gz | wc -l) of $(ls -d $B/sub-* | wc -l)"; grep -c FAILED $OUT/log.txt
