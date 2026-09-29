# Gefäßquerschnitte als Fehlalarme: was die Literatur für unseren Fall hergibt (29.09.2026; Unteragent sonnet, 10 Websuchen; Zahlen vor dem Zitieren an der Quelle prüfen)

Randbedingungen: nur Magnitude (T2*/SWI), keine Phase, 2-5 mm Schichten auf 0,5 x 0,5 x 1 mm umgerechnet. Schon gemessen und
nicht hilfreich: Laufstrecke entlang z (nur Momeni), Frangi (kehrt sich um), harte Negative online (±0), naive künstliche
Verwechsler (Gauß-Röhren: beide Modelle 0,00), Übermalung (löscht Blutungen).

| # | Verfahren | Quelle | Daten | Zahl | Übertragbar? |
|---|---|---|---|---|---|
| 1 | **Physik-geleitete Synthese** von Gefäß-/Kalk-Verwechslern und Blutungen (Dipolkern-Rendering, zufällige Feldstärke 1,5-3 T, TE 20-40 ms) + Zentrumskarte | CenSynCMB, arXiv 2607.05325 (2026) | VALDO T2* (72), AIBL SWI (370, 1,75 mm) | Ablation VALDO: F1 71,7 -> 74,3, Precision 74,6 -> 79,3, **FP je Fall 1,4 -> 0,8** durch die Synthese | JA, gleicher Datensatz. Unterschied zu unserem Arm 1: ihre Verwechsler sind PHYSIKALISCH gerendert (Dipolmuster, Blooming), unsere waren glatte Röhren, die kein Netz verwechselt hat |
| 2 | Geometrische Vorfilterung (Aspektverhältnis = Gefäß) vor zwei 3D-ResNets | "geometric-to-neural cascade", PMC13483767 | 30 SWI, 11 Scanner | FP 1157 -> 269 (-77 %), Sens 93 -> 71 %, F1 0,69 | NEIN: Aspektverhältnis ist genau das Merkmal, das auf unserem Gitter umkippt |
| 3 | Verwechsler-Suche mit YOLO, harte FP als Zusatzklasse fürs 3D-CNN | PMC12410576 | ADNI GRE (1517), AIBL SWI, MAS/OATS SWI | AIBL FP je Scan ~9 -> 1,2; auf ADNI aber F1 0,930 -> 0,902 | teils; entspricht unseren harten Negativen (±0) -- kohortenabhängig |
| 4 | Multi-Task-Lehrer mit **Anatomie-Bedingungen** | Front. Neuroinform. 2023, doi 10.3389/fninf.2023.1204186 | u. a. **OXVASC 2D-T2* 5 mm (74)** | OXVASC: TPR 90 %, **0,9 FP je Fall, Precision 84 %** | RELEVANTESTER Referenzwert für dicke Schichten; lesen, welche Anatomie-Bedingungen (über unsere Liquor-Regel hinaus) |
| 5 | DEEPMIR: T2w + SWI + QSM gemeinsam | Rashid 2021, Sci Rep 11:14124 | MESA, 24 | Sens ~0,8, Prec ~0,6 | nein (QSM Pflicht) |
| 6 | 2,5D FRST-Kandidaten + V-Net, SWI + Hochpass-Phase | Liu 2019, NeuroImage: Clinical | SWI + Phase | Sens 95,8 %, Prec 70,9 %, 1,6 FP je Fall | nur mit Phase -- der obere Vergleichswert |
| 7 | Gelernte Gefäßsegmentierung als Zusatzkanal (VesselBoost o. ä.) | -- | -- | **kein Beleg mit Zahl gefunden**; SHIVA-CMB nennt Vesselness/CSF-Masken nur qualitativ | offene Lücke, ungetestet |
| 8 | Super-Resolution entlang z | arXiv 2608.06311 (WMH), PubMed 42155350 (SWI) | -- | keine CMB-Zahl; Warnung: SR kann kleine Läsionen löschen | nein |
| 9 | Magnitude gegen QSM | ISMRM 2020 #3223 (unsicher) | -- | SWI-Magnitude findet 2,5x so viele CMB wie QSM; nur 15 % der CMB < 5 mm3 auf QSM sichtbar | Phase ist kein Selbstläufer für KLEINE Blutungen |

## Einordnung gegen unsere Messungen
- Der einzige belegte, nachbaubare Hebel ohne Phase ist Nr. 1 -- und zwar in der Fassung, die wir NICHT gebaut haben: Verwechsler
  mit physikalischem Rendering (Suszeptibilitätsquelle -> Dipolfeld -> Signalverlust bei TE, mit Blooming). Unsere Gauß-Röhren wurden
  von beiden Modellen mit 0,00 bewertet, also nie verwechselt; die Dipol-gerenderten von CenSynCMB senkten die Fehlalarme fast um
  die Hälfte. Aufwand: Vorwärtsmodell (Dipolkern-Faltung, TE/B0-Randomisierung, Umrechnung auf unser Gitter mit Schichtdicke), 2-3 Tage;
  danach dieselbe Messvorschrift wie 5ae-1b. Risiko: ihr Gewinn kam zusammen mit T1+T2+T2* und Zentrumskarte; bei uns half die
  Zentrumskarte nicht.
- Nr. 4 ist der einzige Wert auf 5-mm-T2* -- Precision 0,84 bei 0,9 FP je Fall. Wenn das mit "Anatomie-Bedingungen" ohne Phase
  erreicht wurde, ist das unsere Messlatte für dicke Schichten (unsere T2*: Precision 0,70, 3,0 FP je Fall). Paper lesen.
- Nr. 7 (gelernte Gefäßmaske aus Magnitude als Kanal) ist unbelegt, aber die naheliegende Idee mit "anderer Information"; braucht
  ein Gefäßsegmentierungsnetz für SWI/T2* (vortrainiert oder auf wenigen Fällen), Aufwand mittel, Ausgang offen.
- Nr. 9 relativiert den Phasen-Wunsch: für kleine Blutungen ist Magnitude ohnehin empfindlicher; Phase hilft der Precision.

## Empfehlung (Reihenfolge)
1. Paper Nr. 4 und Nr. 1 im Volltext lesen (Anatomie-Bedingungen; Rendering-Details der Verwechsler). Kein GPU.
2. Falls Nr. 1 nachbaubar: Verwechsler-Synthese mit Dipol-Rendering als Arm 1c, Regel wie 5ae-1b, zwei Startwerte.
3. Nr. 7 als Arm 7 nur, wenn ein brauchbares Gefäßsegmentierungsnetz gefunden wird.
Nicht: Aspektverhältnis-Filter, Super-Resolution.

## Nachtrag 29.09. mittags: die zwei Volltexte (Unteragent; WebFetch-Extraktion, Zitate vor dem Übernehmen am PDF prüfen)

**CenSynCMB (arXiv 2607.05325v1, He, ..., Sudre):** Rendering = Dipolkern D(k) = 1/3 - kz^2/|k|^2, statische Dephasierung
phi = gamma * dBz * TE, Magnitude = Betrag des voxelgemittelten komplexen Signals; B0 1,5-3 T und TE 20-40 ms randomisiert; Gefäß-
Verwechsler = "elongated anisotropic masks", Kalk = "compact low-signal sources"; Verwechsler nur als Label 0, keine eigene Klasse;
Positive aus den Prioren der Trainingsfalte (Zahl je Proband, Zentren, Durchmesser, lokale Intensitätsänderung). NICHT angegeben:
Suszeptibilität in ppm, Anzahl je Bild, additiv/ersetzend, Anatomie-Prior, Teilvolumen der Schichtdicke (alles auf 1 mm isotrop
umgerechnet), Code. **Ablation Tabelle IV (VALDO, n = 72, T1+T2+T2*): real 71,7 / +CMB 73,1 / +CMB+Verwechsler 74,3 F1; Precision
74,6 -> 79,3; FP je Fall 1,4 -> 0,8 -- "all p > 0.05".** Folge: der Dipol-Nachbau (Arm 1c) rutscht nach hinten -- 2-3 Tage für einen
Effekt, der in der Quelle selbst nicht gesichert ist, und mit zwei Sequenzen mehr als bei uns.

**Sundaresan et al. 2023, Front. Neuroinform. (Wissensdestillation; Code github.com/v-sundaresan/microbleed-detection = unser
Original!):** Kandidaten-U-Net (Bild + FRST) -> Lehrer (Merkmale + Segmentierer + Klassifikator) -> Schüler per Destillation.
Vorverarbeitung mit Gefäß-Übermalung (bei uns -0,136). **Nachbearbeitung (2.4): Volumen < 2,5 mm3 weg; Elliptizität > 0,2 weg
(Röhren); Kandidaten < 5 mm vom Hirnmaskenrand weg (Sulci).** OXVASC (74 Fälle, 366 CMB, 0,9 x 0,8 x 5 mm, NUR Magnitude):
Cluster-TPR 0,93 -> 0,91 -> 0,90, FP je Fall 195,7 -> 10,1 -> **0,9**, Precision 0,02 -> 0,29 -> **0,84** über die drei Stufen.
UKBB-Ablation: Nachbearbeitung FP 14,7 -> ~0,5 bei TPR 0,90 -> 0,83. Folge: die **Randregel (>= 5 mm vom Hirnrand)** ist bei uns
ungemessen und billig -- Messung am 29.09. mittags (`border_*.json`); die Elliptizität kippt auf unserem Gitter (Frangi-Befund).
Vorsicht beim Vergleich der TPR (Cluster-Regel, andere Kohorte, Median 3 CMB je Fall).
