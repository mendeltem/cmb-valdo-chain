#!/bin/sh
# wk7: alle dicken Vorhersagen parallel (4 Worker x 6 Threads), dazu die
# vorhandenen nativen Karten (Trainingslaeufe) in die Ausgabe kopieren.
# POSIX-sh-kompatibel (hintergrund-Runner).
set -u
cd "${CMB_ROOT:-$(dirname "$(readlink -f "$0")")/..}"
PY=${CMB_PYTHON:-python}
OUT=ergebnisse/schichtdicke/proba

modpfad() {
  case "$1" in
    base) echo ergebnisse/turnier/a03-aniso-e60-gd ;;
    s42)  echo ergebnisse/turnier_hpc_wk7/s42/a03-aniso-e60-gd-augs ;;
    s43)  echo ergebnisse/turnier_hpc_wk7/s43/a03-aniso-s43-e60-gd-augs ;;
  esac
}

# 1) native Karten kopieren (vorhanden aus den Trainingslaeufen, out-of-fold)
for name in base s42 s43; do
  m=$(modpfad $name)
  $PY -c "import json; [print(e['id'], e['fold']) for e in json.load(open('dev/manifest_valdo_gitter_d.json')) if e['cohort']=='valdo-2']" |
  while read -r s k; do
    d=$OUT/$name/native/fold$k
    mkdir -p $d
    cp -n $m/fold$k/${s}_pred_proba.nii.gz $d/
  done
done
echo "native kopiert: $(find $OUT -path '*native*' -name '*pred_proba*' | wc -l) Karten"

# 2) dicke Gitter: 3 Modelle x 3 Dicken x 27 Faelle = 243 Jobs (wiederaufnehmbar)
: > /tmp/wk7_jobs.txt
for name in base s42 s43; do
  m=$(modpfad $name)
  for dicke in n3 n5 n7; do
    case $dicke in
      n3) man=dev/manifest_valdo_gitter_thick_3.json ;;
      n5) man=dev/manifest_valdo_gitter_thick_5.json ;;
      n7) man=dev/manifest_valdo_gitter_thick_7.json ;;
    esac
    for s in $($PY -c "import json; [print(e['id']) for e in json.load(open('$man'))]"); do
      echo "$name $dicke $man $s $m $OUT" >> /tmp/wk7_jobs.txt
    done
  done
done
wc -l /tmp/wk7_jobs.txt
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3
xargs -P 4 -n 6 $PY code/wk7_predict_worker.py < /tmp/wk7_jobs.txt > /tmp/wk7_predict.log 2>&1
echo "fertig, rc=$?"
grep -c "kacheln=" /tmp/wk7_predict.log
find $OUT -name '*pred_proba*' | wc -l
