#!/bin/bash
# we4 CV5 Kette -- wird vom GPU job queue-Antreiber am Rundenende abgearbeitet.
# Nacheinander, EIN GPU-Lauf zur Zeit:
#   1. Probe   (cv5.py --probe -> dev/cv5_probe.json mit schatz_falte_s)
#   2. Vollauf (alle Konfigurationen x 5 Falten, wiederaufnehmbar via dev/cv5_status.json)
#              --zeit wird aus der Probe hochgerechnet (15 Falten + 25% Puffer, min 14 h)
#   3. Aggregation (code/cv5_agg.py: prueft alles, schreibt dann; Sieger -> dev/modelle/best/)
# Logs: logs/cv5_runde15_{probe,voll,agg}.log; Verlauf: logs/cv5_runde15_kette.log
#
# Rueckgabewert (Stand 11.09. Claude; vorher immer 'exit 0'):
#   0          Probe, Vollauf und Aggregation ok
#   1..3       Rueckgabe von cv5_agg.py (1 nichts vollstaendig, 2 Integritaet, 3 Auswahl/Modelle)
#   10+rc      Aggregation ok, aber Vollauf endete mit rc (cv5.py: 1 Faltenfehler, 3 Zeitlimit)
#   20+rc      Probe fehlgeschlagen (Vollauf und Aggregation NICHT gestartet)
#   30         Zeit-Hochrechnung aus der Probe unlesbar
# Der Endwert steht zusaetzlich in logs/cv5_kette_rc.txt (eine Zeile, Zahl; 'laeuft' waehrend
# des Laufs) -- darauf prueft die Abnahme we4.
set -u
cd "${CMB_ROOT:-$(dirname "$(readlink -f "$0")")/..}"
P=${CMB_PYTHON:-python}
mkdir -p logs
RCF=logs/cv5_kette_rc.txt
echo laeuft > "$RCF.tmp" && mv "$RCF.tmp" "$RCF"

ende() {   # $1 = Endwert
    echo "=== KETTE cv5 fertig: $(date '+%F %H:%M:%S') (probe=${RC_PROBE:--} voll=${RC_VOLL:--} agg=${RC_AGG:--}) rc=$1 ===" \
        | tee -a logs/cv5_runde15_kette.log
    echo "$1" > "$RCF.tmp" && mv "$RCF.tmp" "$RCF"
    exit "$1"
}

echo "=== KETTE cv5 start: $(date '+%F %H:%M:%S') ===" | tee -a logs/cv5_runde15_kette.log

$P code/cv5.py --probe > logs/cv5_runde15_probe.log 2>&1
RC_PROBE=$?
echo "probe rc=$RC_PROBE $(date '+%H:%M:%S')" | tee -a logs/cv5_runde15_kette.log
if [ $RC_PROBE -ne 0 ]; then
  echo "PROBE fehlgeschlagen -- Abbruch der Kette, Vollauf NICHT gestartet." | tee -a logs/cv5_runde15_kette.log
  ende $((20 + RC_PROBE))
fi

# Zeit fuer den Vollauf aus der Probe hochrechnen (15 Falten, 25% Puffer, min 14 h).
ZEIT=$($P - <<'PY'
import json
z = json.load(open('dev/cv5_probe.json'))
f = z.get('schatz_falte_s', 3400)
print(max(int(f * 15 * 1.25), 50400))
PY
)
if ! [[ "$ZEIT" =~ ^[0-9]+$ ]]; then
  echo "Hochrechnung aus dev/cv5_probe.json unlesbar ('$ZEIT') -- Abbruch." | tee -a logs/cv5_runde15_kette.log
  ende 30
fi
echo "Vollauf --zeit $ZEIT s (Hochrechnung aus Probe)" | tee -a logs/cv5_runde15_kette.log

$P code/cv5.py --zeit "$ZEIT" > logs/cv5_runde15_voll.log 2>&1
RC_VOLL=$?
echo "voll rc=$RC_VOLL $(date '+%H:%M:%S')" | tee -a logs/cv5_runde15_kette.log

# Aggregation laeuft auch nach einem Teilausfall: sie meldet unvollstaendige Konfigurationen
# (ergebnisse/cv5_agg_bericht.json) und schreibt summary.json nur, wenn alles stimmt.
CUDA_VISIBLE_DEVICES= $P code/cv5_agg.py > logs/cv5_runde15_agg.log 2>&1
RC_AGG=$?
echo "agg rc=$RC_AGG $(date '+%H:%M:%S')" | tee -a logs/cv5_runde15_kette.log

if [ $RC_AGG -ne 0 ]; then ende $RC_AGG; fi
if [ $RC_VOLL -ne 0 ]; then ende $((10 + RC_VOLL)); fi
ende 0
