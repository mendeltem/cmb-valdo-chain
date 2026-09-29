# Wofür die Zahl der Mikroblutungen klinisch gebraucht wird -- und was das für unsere Kette heißt (29.09.2026; Unteragent sonnet, 8 Suchen; Zitate vor Gebrauch prüfen)

| # | Klinischer Gebrauch | Schwelle | Quelle | Für die Kette (Sens 0,65 / Prec 0,65 / Fall-Genauigkeit 0,7 am F1-Optimum) |
|---|---|---|---|---|
| 1 | Boston-Kriterien 2.0, CAA | >= 2 streng lobäre hämorrhagische Läsionen (Sens 64,5 %, Spez 95 %) | Charidimou 2022, Lancet Neurol, PMID 35841910 | binär 0 / 1 / >= 2 -- ein Fehlalarm hebt einen Fall über die Schwelle |
| 2 | Antikoagulation bei VHF (CROMIS-2) | Bürdeklassen 0 / 1-4 / >= 5; ICH-Rate 9,8 gegen 2,6 je 1000 Patientenjahre, HR 3,67 | Wilson/Charidimou 2018, Lancet Neurol, PMC5956310 | grobe Klassen, nicht die exakte Zahl; Fall-Genauigkeit 0,7 reicht nicht allein |
| 3 | Thrombolyse (ESO 2021) | > 10 CMB: Empfehlung gegen i.v.-Lyse (schwach, niedrige Evidenz) | ESO 2021, PMC7995316 | Überzählen kann Therapie verweigern |
| 4 | Anti-Amyloid-Antikörper (ARIA-H) | > 4 CMB = Ausschluss (Lecanemab/Donanemab-Studien, Appropriate-Use) | FDA-Label Leqembi 2023; ARIA-Übersicht PMC12660453 | schärfste Schwelle: 4 gegen 5 |
| 5 | CAD-Zielwerte (Vorsortierung) | > 90 % Sensitivität bei < 5 FP je Fall gilt als akzeptabel; 3D-CNN 95,8 % / 1,6 FP mit Phase | CAD-Übersicht; Liu 2019 | unser F1-Arbeitspunkt (0,65) ist KEIN Vorsortier-Arbeitspunkt |
| 6 | Bürdeklasse je Patient statt Läsionszahl | 2D-T2*-Modell: Läsions-F1 0,70; Klasse >= 4: Sens 0,93 / Spez 0,94 | Front. Aging Neurosci. 2026, 10.3389/fnagi.2026.1729422 | so sollte auch unsere Auslieferung gemessen werden |
| 7 | Prävalenz | Rotterdam 18,7 %, Median 1 CMB; Schlaganfallkohorte 49 % | PMC3881231, PMC13239722 | bei Median 1 kippt ein einziger Fehlalarm die Klasse |
| 8 | expliziter "tolerabler Zählfehler" | nicht definiert; Akzeptanz über Sensitivität/FP-Ziele oder Klassengrenzen | -- | Zählfehler je Fall trotzdem berichten |

## Folgerungen
1. **"Sensitivität vor Precision" ist für den Einsatz richtig** (Vorsortierung, Befunder verwirft Fehlalarme; Übersehenes geht verloren) -- aber der
   Maßstab dafür ist > 90 % Sensitivität, nicht 0,65. Unser F1-Optimum ist die Zahl fürs Paper, nicht der Arbeitspunkt für die Klinik.
   Wir haben den Vorsortier-Arbeitspunkt schon gemessen, nur nicht so genannt: Kandidatenschwelle 0,15 erreicht 103 von 139 VALDO-Blutungen
   (0,74) bei ~4,9 Kandidaten je Fall (5w); Vereinigung dreier Startwerte 120 von 139 (0,86). Als eigene Zeile "Vorsortier-Modus" berichten.
2. **Bürdeklasse je Patient als klinisches Maß** (0 / 1-4 / >= 5 und die ARIA-Schwelle > 4): Sensitivität/Spezifität der Klasse aus unseren
   vorhandenen Vorhersagen berechnen (Fall-Zahlen liegen in `evaluate`). Das ist die Zahl, die ein Kliniker versteht -- und die Frontiers-2026-Arbeit
   zeigt, dass ein Läsions-F1 um 0,70 dort 0,93 / 0,94 ergeben kann.
3. Keine autonome Entscheidung: alle vier gefundenen Schwellenentscheidungen brauchen die manuelle Nachzählung; so ins Paper (Discussion, Grenzen).
