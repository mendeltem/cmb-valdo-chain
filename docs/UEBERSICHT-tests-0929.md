# Übersicht aller Tests der CMB-Linie (Stand 29.09.2026, 08:15) und die beste Kette für catalina

Alle Zahlen sind Läsions-F1 (gepoolt, 26er-Nachbarschaft, Berührungsregel), gepaart über Fälle mit Bootstrap; "gesichert" =
95-%-Intervall schließt Null aus. VALDO = 57 CV-Fälle, catalina-T2* = 75 Fälle / 828 CMB, catalina-SWI = 61 / 652.
Rauschmaß: Einzelstartwert des U-Nets ~0.08 (drei Startwerte 0.646 / 0.590 / 0.578), zweite Stufe sd 0.003-0.009 (10 Startwerte).
Quellen: ARBEIT.md (Abschnittsnummern in Klammern), mb-arena/protokoll.md, REPLIKATION.md.

## 1 Was gemessen wurde

### Datenkette / Vorverarbeitung (VALDO, gepaart gegen die jeweilige Basis)
| Hebel | Ergebnis | Urteil |
|---|---|---|
| Gefäß-Übermalung des Originals weglassen (4.1) | **+0.136 gesichert** | übernommen |
| FRST als zweiter Kanal behalten (4.11) | **+0.053** (0.588 gegen 0.535) | übernommen |
| Umrechnung z = 1 mm statt 2 mm (4.8) | +0.036 | übernommen |
| Bias-Korrektur (FSL FAST -B) behalten (4.9) | +0.022 n. s. | übernommen |
| Perzentil-Skala p99.5 statt Maximum (4.9) | +0.025 n. s., ein Startwert | nicht übernommen |
| z-Normierung je Fall (5ac) | VALDO -0.005, Zero-shot T2* -0.004 / SWI -0.022, alle n. s. | nicht übernommen |
| Landmarken-Angleich Nyul-Udupa (5ac) | VALDO -0.064, Zero-shot **-0.048 / -0.034 gesichert**, gemischt VALDO-Anteil 0.355 | nicht übernommen |
| kubische statt lineare Interpolation (5ae Arm 2) | Ensemble -0.024 n. s.; Sensitivität klein 0.50 -> 0.68, FP 0.82 -> 1.25 | nicht übernommen |
| Gewebekarte (SynthSeg) als Eingabekanäle (5m) | -0.058 | nicht übernommen |
| SynthSeg-Maske statt VALDO-Maske (`--maske synthseg`) | für Kohorte 3 nötig (Kopfmaske), sonst gleich | Werkzeug |

### Erste Stufe: Architektur und Training (VALDO)
| Hebel | Ergebnis | Urteil |
|---|---|---|
| Turnier 9 Netze, 40 Ep. (4.4) | a03-aniso 0.404 roh gegen Attention-U-Net 0.245: **+0.159 gesichert**; swin, resse, segres, flach gesichert schlechter | a03-aniso |
| nnU-Net / Attention-U-Net auf sauberen Daten, 60 Ep. (5s) | 0.458 / 0.463 gegen 0.585 roh | a03-aniso bleibt |
| Epochen 60 gegen 120 (5h) | VALDO -0.029 roh, catalina-T2* -0.022 gesichert | 60 Ep. |
| letzter Zwischenstand statt Wahl auf innerer Falte (5o) | Mittel sinkt 0.618 -> 0.604 | Zwischenstandswahl bleibt |
| Ausschnitt 24 / Rezept / 48 mm (5t) | 0.369 / 0.585 / 0.569 (24 mm -0.216 gesichert) | 38 x 31 x 47 mm |
| Verhältnis positiv:negativ 1:2 (5g) | kollabiert (leere Vorhersage) | 60/25/15 bleibt |
| Augmentierung Drehung/Maßstab/Kontrast (4.7, 5p) | Dickschicht -0.127, Intensität -0.108; Zero-shot ebenfalls negativ | nur Spiegelung |
| blob loss (5j) | -0.044 n. s.; catalina-T2* -0.029 gesichert; SWI -0.003 (kleine CMB 0.37 -> 0.44) | nein |
| Zentrumskarte als Hilfsausgabe (5k) | -0.054 gesichert | nein |
| fnrw-Verlust (5l) | -0.030 n. s. | nein |
| harte Negative online (5l) | +0.000 | nein |
| künstliche Blutungen + Verwechsler, 1. Fassung (5ae Arm 1) | Ensemble **-0.130 gesichert**; Verwechsler von beiden Modellen mit 0.00 bewertet | nein |
| künstliche Blutungen, Budget erhalten, 2. Fassung (5ae-1b) | Ensemble **-0.072 gesichert**; Sensitivität klein 0.50 -> 0.57, FP 0.82 -> 1.21 | nein, Arm geschlossen |
| Ensemble über Startwerte (5e, 5u) | ~ bestes Mitglied, +0.02-0.04 über dem Mittel, flach ab 2 | **2 Startwerte ausliefern** |
| Spiegel-TTA (5e) | VALDO +0.02, catalina-T2* -0.002 | nein |

### Übertragung auf catalina (erste Stufe)
| Strategie | catalina-T2* | catalina-SWI | Urteil |
|---|---|---|---|
| VALDO-Modell zero-shot | 0.622 (s1 0.638) | 0.305 (s1 0.311) | Vergleichswert |
| eigenes Training 60 Ep. | **0.663** | **0.626** | **Basis** |
| Feintuning vom VALDO-Modell, 30 Ep. | = eigenes Training | eher schlechter | nein |
| gemischt VALDO + catalina, gleiches Budget je Fall (5h) | 0.641 (e120), +0.001 gegen eigenes e120 | -- | nein |
| Vortraining auf allen + Feintuning (5e) | -0.039 gesichert | -0.045 gesichert | nein |
| gemischt mit Landmarken (5ac) | 0.630 | -- | nein |
| zweiter Startwert eigenes Training | -- | s1 0.613 (-0.022 n. s.) | Ensemble sinnvoll, nicht gemessen |

### Zweite Stufe (Kandidaten aus Schwelle 0.15 / 1 mm3, `--keep-csf`)
| Variante | VALDO | catalina-T2* | catalina-SWI | Urteil |
|---|---|---|---|---|
| nur Schwelle (feste Zelle 0.3 / 2 mm3) | 0.585 roh | 0.654 | 0.626 | Bezug |
| + harte Liquor-Regel (4.3) | **0.646** (+0.06, 0 TP verloren) | 0.670 | 0.614 (löscht 74 echte) | VALDO/T2* ja, SWI **nie** |
| je Kohorte: logistisch / Wald / Boosting / CNN (5d) | ≤ Schwelle | ≤ Schwelle | ≤ Schwelle | nein |
| gemeinsamer Wald / Logistik (5e) | ±0.04 mit Pool-Zusammensetzung | -- | Gewinn löste sich mit unabhängiger Stufe 1 auf | nein |
| **gemeinsames 3D-CNN, 16-mm-Würfel** (5e, 10 Startwerte 5z) | 0.632 (= Regel) | 0.671 (= Regel) | **0.672 (+0.058 gegen Regel, +0.045 bestätigt)** | **ja** |
| zweiskalig 16 + 28 mm (5r) | -0.030 gesichert | +0.016 / +0.019 (einmal gesichert) | +0.016 / +0.013 n. s. | nicht übernommen (VALDO-Regel) |
| Lehrer-Schüler alpha 0.3, 10 Startwerte (5z) | +0.001 | +0.003 | +0.002 | nein |
| selbstüberwachtes Vortraining des Encoders (5ab) | +0.012 | -0.005 | -0.011 | nein |
| Vereinigung der Kandidaten dreier Startwerte (5y) | -0.039 | 0.000 | +0.006 | nein |
| Vereinigung nur Mehrheits-Kandidaten (5y) | -0.015 | 0.000 | +0.007 | nein |
| Kandidaten empfindlicherer erster Stufen als Zusatz (5s) | Kette 0.594 gegen 0.625 | -- | -- | nein |
| **VALDO-Klassifikator zero-shot** (5ae Arm 4) | -- | 0.648 (-0.022, Schwelle zu streng) | **0.671 (= Pool)** | veröffentlichbar |
| Feintuning des VALDO-Klassifikators 10 / 25 / 50 / 100 % | -- | -0.064 / -0.025 / -0.004 / 0.000 | -0.051 / -0.021 / -0.009 / -0.012 | erst ab 50 % neutral |

### Externe Validierung und Prüfmenge
| Messung | Ergebnis |
|---|---|
| 15 beiseitegelegte VALDO-Fälle, eingefrorene Kette (5ad) | gepoolt 0.468 (ein Fall mit 73 CMB, S 0.20), ohne ihn 0.731, Median je Fall 0.667 |
| Momeni/CSIRO (57 SWI, 146 Punkte), zero-shot ohne Regel | VALDO-U-Net 0.525, SWI-U-Net 0.491, Sensitivität ~0.8 |
| dito mit Liquor-Regel | 0.613 / 0.555, Median je Proband 0.667; Regel kostet hier Sensitivität 0.80 -> 0.64 |
| dito mit VALDO-Klassifikator (`apply.py`) | VALDO-U-Net 0.607 (= Regel), **SWI-U-Net 0.630 (+0.106 gesichert gegen Regel)** |

### Werkzeuge, die sich als Entscheider NICHT bewährt haben
CPU-Vorprüfung der Intensitätsangleichung (5aa/5ac): Rangfolge vom GPU-Lauf widerlegt. Kandidaten-Obergrenze (5w/5y): höhere Decke
brachte nichts. Robustheitsprobe (5v): einmal bestätigt. Alle drei beschreiben, keiner entscheidet.

## 2 Die beste Kette für catalina (höchstes gemessenes F1)

**Vorverarbeitung (je Fall, beide Sequenzen gleich):** HD-BET-Hirnmaske; FSL FAST -B Bias-Korrektur (einmal); invertierte Skala
1 - I / max im Hirn (keine z-Normierung, keine Landmarken, kein p99.5); KEINE Gefäß-Übermalung; Umrechnung auf 0.5 x 0.5 x 1.0 mm,
linear, Hirnzuschnitt (`code/gitter_c_bauen.py`, `cmb.transfer.build_private_grids`); FRST auf dem Gitter als zweiter Kanal
(`code/frst.py`, Radien 2/3/4/6 Voxel); SynthSeg --robust auf dem nativen biaskorrigierten Bild für die Gewebeklasse (nur für die zweite Stufe).

**Erste Stufe (je Kohorte EIGENES Training, nie gemischt, nie feingetunt):** a03-aniso (anisotropes 3D-U-Net, 22.5 Mio.
Parameter, Eingabe T2*/SWI + FRST), 60 Epochen, Ausschnitte 38 x 31 x 47 mm, 800 je Epoche, Stichprobe 60 % Läsion / 25 % Hirn /
15 % beliebig, nur Spiegelungen, AdamW 3e-4 mit Cosinus, BCE + Dice mit tiefer Überwachung, Zwischenstand auf der inneren Falte
gewählt. Gemessen: T2* 0.663, SWI 0.626 (Schwelle allein). Ausliefern: Mittel der fünf Faltenmodelle, und möglichst zwei
Startwerte gemittelt (auf catalina nicht gemessen; auf VALDO +0.02-0.04 über dem Mittel, und welcher Startwert vorn liegt, ist vorher nicht zu wissen).

**Nachbearbeitung Stufe 1:** Schwelle 0.3, Mindestgröße 2 mm3, Riesenkomponenten weg (feste Zelle; kreuzweise Wahl schwankt um 0.02-0.05).

**Zweite Stufe:** Kandidaten bei Schwelle 0.15 / 1 mm3 mit Gewebeklasse (`cmb.stage2.candidates --keep-csf`); gemeinsames
3D-CNN auf dem Pool aller Kohorten (`cmb.stage2.pooled --models cnn`, 16-mm-Würfel aus Bild + FRST + Wahrscheinlichkeit, drei
Mitglieder gemittelt, Schwelle je Kohorte aus inneren Falten). Gemessen: **T2* 0.671, SWI 0.672**. Auf T2* ist die harte
Liquor-Regel gleichwertig (0.670) und billiger; auf SWI darf die harte Regel NICHT laufen (0.614, löscht 74 echte Blutungen).
Wenn nur catalina zählt: das zweiskalige CNN (16 + 28 mm) liegt auf T2* bei 0.687-0.691 (+0.016 / +0.019, einmal gesichert)
und auf SWI bei 0.678-0.686 (+0.013 / +0.016 n. s.) -- nach der VALDO-Regel verworfen (dort -0.030), für eine reine
catalina-Auslieferung die höchste gemessene Zahl, aber nur mit weiteren Startwerten belastbar.

**Veröffentlichbare Variante (ohne private Kandidaten im Training):** VALDO-Klassifikator zero-shot (`cmb.stage2.apply`):
SWI 0.671 (gleich), T2* 0.648 (-0.022, weil die VALDO-Schwelle 0.50 für T2* zu streng ist -- eine lokal gewählte Schwelle
würde das voraussichtlich schließen, ist aber nicht gemessen).

**Nicht tun:** Mischen oder Feintuning des U-Nets zwischen Kohorten; harte Liquor-Regel auf SWI; z-Normierung / Landmarken;
Augmentierung außer Spiegelung; 120 Epochen; blob / Zentrum / fnrw; TTA auf catalina; künstliche Blutungen.
