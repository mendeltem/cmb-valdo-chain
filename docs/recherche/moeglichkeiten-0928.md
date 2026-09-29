# Welche Möglichkeiten bleiben? (Nutzerfrage 28.09.2026 nach dem Einfrieren der VALDO-Kette)

Maßstab (ARBEIT.md 5q, 5ac): eine Idee zählt nur, wenn sie dem Netz ANDERE Information gibt oder die Trainingsverteilung
verändert. Alles, was das Netz nur empfindlicher macht, ist vermessen und bringt kein F1. Die gemessene Schwäche (5ad):
kleine (< 10 mm3), kontrastarme Blutungen bei massiver Last werden gar nicht gesehen (Höchstwahrscheinlichkeit < 0,05).

## A Literatur (Unteragent sonnet, 8 Websuchen, 28.09.; Zahlen nur aus den Quellen, VOR dem Zitieren zeilenweise gegenlesen)

| # | Verfahren | Quelle | Zahl | Neue Information |
|---|---|---|---|---|
| 1 | Physikbasierte Synthese positiver CMB + gelabelte Verwechsler (Gefäß, Kalk) in reale Hintergründe | CenSynCMB, arXiv 2607.05325 (2026) | VALDO Task 2, Läsions-F1: Attention-U-Net 65,8 -> Zentrumskarte 70,3 -> +FN-Neugewichtung 71,7 -> +Synthese 74,3 (p = 0,020); extern AIBL-SWI Recall 88,5, F1 65,0 | Beispiele für kompakte kontrastarme Läsionen und für die häufigsten Fehlalarmformen, die im Trainingssatz selten sind |
| 2 | Analytische 3D-Gauß-Synthese von CMB, Teilvolumen durch Herunterrechnen | Momeni et al. 2021, PubMed 34051412 | nur synthetisch trainiert, real getestet: ~9 FP je Scan (RF); 37 000 synthetische Läsionen veröffentlicht; keine Sensitivität kleiner CMB im Auszug | Formverteilung ohne reale Referenz |
| 3 | T1 + T2 + T2* als Kanäle, Modelle je Schichtdicke | MixMicrobleed (VALDO-Gewinner), arXiv 2108.02482; Challenge-Paper arXiv 2208.07167 | Balanced Accuracy 87,3 (Gesamt); KEINE Ablation nur für die Kanäle | Anatomie/Kontrast aus T1/T2 |
| 4 | QSM statt Magnitude, FRST-Kandidaten + V-Net | Xia et al., JMRI 2024, doi 10.1002/jmri.29198 (393 Patienten, 1843 CMB) | Sensitivität gesamt 88,9, 2,87 FP je Patient | Dipolmuster der Suszeptibilität (Phase) |
| 5 | QSM gegen SWI, Leserstudie | PubMed 36168880 (2022) | keine Zahl im Auszug | Phase |
| 6 | Wissensgeleitetes 2,5D-CNN | Biomed. Signal Proc. Control 2023 | Sens 98,2, 1,72 FP je Patient, eigener SWI-Satz, keine Ablation | anatomische Priors |
| 7 | mIP als Eingangskanal | nur als Betrachtungshilfe belegt (AJNR 7T) | KEINE Zahl für ein Netz gefunden | Fortsetzung über Schichten |
| 8 | Pseudo-Labels / Selbstüberwachung für CMB | kein CMB-spezifischer Beleg mit Zahl | -- | unmarkierte Bilder |
| 9 | Multi-Source-Training T2*/SWI mehrerer Zentren | SHIVA-CMB, Sci Rep 2024, s41598-024-81870-5 | keine Zahl abgerufen | breitere Verteilung |

Einordnung des Agenten, von mir geprüft: der einzige Eintrag mit Gewinn AUF VALDO und Bezug zu kleinen kontrastarmen
Läsionen ist Nr. 1. Phase/QSM (4, 5) ist die stärkste "andere Information", aber VALDO liefert keine Phase und catalina
hat lokal keine (geprüft 28.09.: keine Phasendateien unter ~/data). mIP als Kanal und Pseudo-Labels sind Lücken, keine
negativen Befunde.

## B Öffentliche Datensätze mit CMB-Markierung außer VALDO (zweiter Unteragent, 8 Websuchen; Lizenz/Parameter VOR Gebrauch an der Quelle prüfen)

| Datensatz | Zugang | Umfang | Markierung | Bewertung |
|---|---|---|---|---|
| Momeni/CSIRO "Synthetic Cerebral Microbleed on SWI images" (researchdata.edu.au, 1702755) | scheinbar frei über CSIRO-Repositorium; Lizenz unklar | 75 SWI mit 175 realen CMB + dieselben Scans mit je 10 synthetischen | Orte/Punkte, keine Voxelmasken belegt | einziger realer Zusatzsatz; für das SWI-Modell (catalina-SWI zero-shot 0,305!) als Trainingszuwachs oder externer Test denkbar |
| Momeni AIBL (57 SWI, 146 CMB-Orte) | AIBL-Antrag, kein Link gefunden | klein | Orte | nur mit Antrag |
| Dou et al. 2016 (320 SWI) | NICHT öffentlich | -- | Punkte | historischer Benchmark, unerreichbar |
| UK Biobank swMRI (~50 000, Magnitude + Phase, 0,8 x 0,8 x 3 mm) | kostenpflichtiger Antrag | riesig | KEINE CMB-Voxelmasken | für Selbstüberwachung/Pseudo-Labels denkbar, nicht für Referenzen |
| Rotterdam (830, 3D-T2*) | nur Kooperation | -- | keine Masken | unerreichbar |
| CenSynCMB | Synthese-Pipeline, kein neuer Patientensatz; Code-Verfügbarkeit unklar | -- | synthetisch | siehe A1 |

Nicht gefunden (Hinweis, keine sichere Verneinung): OASIS, ADNI, HCP, Kaggle, PhysioNet, Synapse, Grand-Challenge außer
VALDO, MRBrainS, chinesische CMB-Sätze 2023-2026.

## C Intern noch offene Ideen (aus ARBEIT.md, nie gemessen)

- **Kubische Interpolation / Training in nativer Auflösung (4.8):** die Umrechnung auf 0,5 x 0,5 x 1 mm kostet gemessen 15-30 % Läsionskontrast, am meisten in den dickschichtigen Kohorten -- sub-110 ist Kohorte 1 (4 mm) und scheitert an kontrastarmen Läsionen. z = 2 mm war negativ (-0,036 n. s.), kubisch ist ungemessen. Achtung nach 5ac: eine CPU-Kontrastmessung DARF diesen Lauf nicht vorentscheiden.
- **T1 + T2 als Kanäle (5q e):** Obergrenze der Ein-Sequenz-Lösung, nur VALDO; catalina hat kein T1.
- **Künstliche Blutungen und Verwechsler (5q c):** Plan war "nur für den Klassifikator"; A1 zeigt den Gewinn aber in der ERSTEN Stufe (Sensitivität), weil dort die Beispiele fehlen.
- **mIP als Kanal (5q b):** billig (Minimum über ±2 Schichten auf dem Gitter), ohne Literaturbeleg; ein 3D-Netz sieht den z-Kontext schon.
- **Phase:** VALDO keine, catalina lokal keine (geprüft 28.09.). Nur über PACS-Export nachträglich zu beschaffen.
- **Auslieferung auf allen 72 VALDO-Fällen nachtrainieren:** +26 % Trainingsdaten, nicht mehr messbar (die 15 sind verbraucht); üblich und sinnvoll für das ausgelieferte Modell, aber die Paper-Zahl bleibt die der CV/Prüfmenge.

## D Rangfolge (Ertrag je Aufwand, Stand 28.09.)

1. **Künstliche kleine, kontrastarme Blutungen + künstliche Verwechsler in die ERSTE Stufe** (A1). Einziger Hebel mit Gewinn AUF VALDO (+2,6 Punkte über den anderen Hebeln, insgesamt 65,8 -> 74,3) und der einzige, der die gemessene Schwäche trifft: das Netz hat im Training fast keine Beispiele für massive Last kleiner kontrastarmer Blutungen (Kohorte 1: drei CV-Fälle über 5). Neue Information = Verwechsler-Beispiele; das unterscheidet ihn von blob/Zentrum, die nur empfindlicher machten. Aufwand: mehrere Tage Code (Gauß-Läsion nach Momeni, Kontrast aus der eigenen Kontrastverteilung gezogen, Ort in weißer Substanz/Kortex per SynthSeg; Verwechsler als kurze dunkle Zylinder = Gefäßquerschnitte), zwei Trainings mit zwei Startwerten. Messen NUR in der CV, gepaart, Regel vorab: Gewinn zählt erst über 0,08 (Rauschmaß) oder mit zwei Startwerten übereinstimmend.
2. **Kubische Interpolation des Bildes** (C): ein Gitter neu rechnen, ein Training; direkte Antwort auf "Kontrastverlust durch Umrechnung". Klein, aber billig.
3. **Momeni/CSIRO-SWI-Satz beschaffen** (B): Lizenz und Bildparameter prüfen; wenn frei, als externer Test des SWI-Modells (erste unabhängige öffentliche Zahl) und als Trainingszuwachs für SWI. Kein GPU-Aufwand für die Prüfung.
4. **T1/T2-Obergrenze** (nur Paper-Zahl, 2 GPU-Stunden).
5. **mIP-Kanal** (ein Training, geringe Erwartung).
6. **Phase/QSM**: stärkste andere Information (A4), aber ohne Daten. Für die nächste Datenerhebung vormerken.

Nicht empfohlen: Pseudo-Labels (kein CMB-Beleg), Foundation-Modelle (keine Zahl für Läsionen dieser Größe), weitere
Architekturen (5s), weitere Varianten der zweiten Stufe (5r, 5z, 5ab, 5y).
Vorbedingung für alles: das verblindete Expertenurteil (5b) -- sind viele "Fehlalarme" echte Blutungen, ist die Precision
unterschätzt und die Rangfolge oben verschiebt sich zur Sensitivität hin.
