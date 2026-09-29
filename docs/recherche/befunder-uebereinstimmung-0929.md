# Wie gut stimmen Befunder bei Mikroblutungen überein? Obergrenze für 0,65 (29.09.2026; Unteragent sonnet, 8 Suchen; Zitate vor Gebrauch prüfen)

| # | Befund | Zahl | Quelle |
|---|---|---|---|
| 1 | MARS, Inter-Rater "CMB vorhanden" | Kappa 0,68 [0,58-0,78] bei TE 40 ms, bis 0,87 bei TE 26 ms (sichere CMB) | Gregoire 2009, Neurology 73:1759 |
| 2 | MARS, Anzahl | ICC intra 0,98, inter 0,93 | dieselbe |
| 3 | BOMBS, ">= 1 CMB" | Kappa 0,68 [0,49-0,86] beliebig, 0,78 lobär | Cordonnier 2009, JNNP, PMID 19008468 |
| 4 | Dickschicht-GRE gegen Dünnschicht-SWI | nur 33 % (103/310) der SWI-CMB auf GRE sichtbar | Nandigam 2009, AJNR 30:338 |
| 5 | T2*-GRE gegen SWI, n = 246 | 365 gegen 699 CMB; CAA-Fälle 1146 gegen 1432 | Shams 2015, AJNR 36:1089 |
| 6 | bestes SWI-Modell gegen erfahrensten Befunder | Sens 95,8 %, Prec 70,9 %, 1,6 FP je Fall (mit Phase) | Liu 2019, NeuroImage: Clinical |
| 7 | Anteil "möglicher" CMB | 16,7 % der Kandidaten strittig; 53,8 % davon bei Nachuntersuchung verschwunden | Neuroradiology 2021, PMC9117345 |
| 8 | VALDO-Veranstalter | Referenz unsicher, starke Inter-/Intra-Rater-Variabilität, keine sichere Ground Truth | Sudre 2024, Med Image Anal 91:103029 |
Nicht gefunden: eine Mensch-gegen-Mensch-F1- oder Sensitivitätszahl auf Läsionsebene (Dou 2016, Momeni, CenSynCMB nennen es nur qualitativ).

## Für uns
- 0,65 liegt im Band der dokumentierten Mensch-Mensch-Übereinstimmung (Kappa 0,68-0,87), auch wenn Kappa und F1 nicht dasselbe messen.
  Die Obergrenze ist kein fester Wert: Sequenz und Schichtdicke ändern die gefundene Zahl um Faktor 2-3, ein Sechstel der Kandidaten ist
  per Definition strittig. Plausibles Band der Referenz: 0,65-0,85. **Unsere Zahl liegt am unteren Rand dieses Bands, nicht darunter.**
- Eine härtere Aussage braucht die Zweitannotation auf den eigenen Daten -- genau das ist die verblindete Prüfseite vom 22.09. (215
  Ansichten, beim Nutzer). Sie ist damit nicht nur eine Kontrolle, sondern die Zahl, die der Literatur fehlt.
- Ins Paper (Diskussion): Nandigam/Shams als Beleg, dass der Kohortensprung (T2* 5 mm gegen SWI 2 mm) auch eine Referenzfrage ist;
  Punkt 7 als Grund, "mögliche" CMB in der eigenen Kohorte getrennt zu markieren.
