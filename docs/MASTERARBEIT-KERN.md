# Kern der Masterarbeit: von MicrobleedNet zu unserer Kette (Entwurf 29.09.2026, Stand der Zahlen: ARBEIT.md 5ae)

Leitgedanke: Wir haben MicrobleedNet (Sundaresan et al. 2023) auf VALDO nachgebaut und dann JEDEN Baustein einzeln gegen
eine gepaarte Messung getauscht. Nur was gesichert oder reproduziert besser war, blieb. Die Arbeit erzählt diese Kette,
nicht die 40 Versuche, die nichts brachten (die stehen als eine Tabelle im Anhang).

Metrik überall: Läsions-F1 (gepoolt, 26er-Nachbarschaft, Berührungsregel), 5-fach-Kreuzvalidierung auf 57 VALDO-Fällen
(139 Blutungen), gepaarter Bootstrap über Fälle; "gesichert" = 95-%-Intervall ohne Null. Rauschmaß eines einzelnen
Startwerts ~0.08, deshalb zwei Startwerte je Entscheidung.

## 0 Abstract (Entwurf 29.09.2026; Zahlen vor Abgabe gegen ARBEIT.md prüfen)

**Deutsch.** Zerebrale Mikroblutungen (CMB) sind wenige Millimeter große, hypointense Läsionen auf T2*- und suszeptibilitätsgewichteten
Aufnahmen, deren manuelle Zählung aufwendig und wenig reproduzierbar ist. Ausgangspunkt dieser Arbeit ist MicrobleedNet
(Sundaresan et al. 2023), ein zweistufiges, öffentlich verfügbares Verfahren. Wir haben es auf dem öffentlichen VALDO-Datensatz
(72 Fälle, drei Kohorten mit 0,8 bis 4 mm Schichtdicke) nachgebaut und anschließend jeden Baustein der Kette einzeln gegen eine
gepaarte, über Fälle gebootstrappte Messung in 5-fach-Kreuzvalidierung getauscht; nur Änderungen, die gesichert oder mit
zweitem Startwert reproduziert besser waren, blieben. Sieben Änderungen trugen: der Verzicht auf die Gefäß-Übermalung des Originals
(+0,14), ein anisotropes 3D-U-Net statt eines isotropen Attention-U-Nets (+0,16), ein gemeinsames Trainingsgitter von 0,5 x 0,5 x 1 mm,
der beibehaltene FRST-Kanal (+0,05), eine feste Stichprobe mit begrenztem Budget je Fall, eine anatomische Nachbearbeitung über
SynthSeg-Gewebekarten (Liquor-Regel, +0,06 ohne Verlust echter Blutungen) und ein kohortenübergreifend trainierter 3D-CNN-Klassifikator
als zweite Stufe. Das Läsions-F1 stieg von 0,245 auf 0,647; auf zwei eigenen Klinik-Kohorten (T2*, 75 Fälle; SWI, 61 Fälle) erreichte
die Kette mit kohortenspezifischem U-Net und gemeinsamem Klassifikator 0,67, gegenüber 0,41 für das auf denselben Falten feingetunte
Original. Auf einer fremden öffentlichen SWI-Kohorte (57 Fälle) lag die rein auf VALDO trainierte Kette ohne Anpassung bei 0,61 bis 0,65
(Median je Proband 0,67). Rund vierzig weitere Eingriffe -- Verlustfunktionen, Augmentierung, Intensitätsangleichung, künstliche
Läsionen, längeres Training, Varianten der zweiten Stufe, Formfilter gegen Gefäße -- erhöhten die Sensitivität, nicht das F1: der
verbleibende Engpass ist die Trennung von Blutung und Gefäßquerschnitt mit einer einzigen Magnitudensequenz, besonders bei massiver
Last kleiner, kontrastarmer Blutungen. Die resultierende Kette ist vollständig aus öffentlichen Daten trainierbar und wird mit Code,
Befehlen und allen negativen Ergebnissen veröffentlicht.

**English.** Cerebral microbleeds (CMB) are millimetre-sized hypointense lesions on T2*-weighted and susceptibility-weighted MRI whose
manual counting is laborious and poorly reproducible. Starting from MicrobleedNet (Sundaresan et al. 2023), a public two-stage
method, we re-implemented the pipeline on the public VALDO dataset (72 cases, three cohorts, 0.8-4 mm slices) and then replaced
each component in turn, accepting a change only if it improved the paired, case-bootstrapped lesion-level F1 in 5-fold cross-validation
and survived a second random seed. Seven changes held: dropping the original's vessel inpainting (+0.14), an anisotropic 3D U-Net
instead of an isotropic attention U-Net (+0.16), a common 0.5 x 0.5 x 1 mm training grid, the retained radial-symmetry channel (+0.05),
a fixed sampling scheme with a bounded budget per case, an anatomical post-processing rule from SynthSeg tissue maps (CSF rule, +0.06
with no true lesion lost) and a cross-cohort 3D-CNN candidate classifier as second stage. Lesion F1 rose from 0.245 to 0.647; on two
in-house clinical cohorts (T2*, 75 cases; SWI, 61 cases) the chain with cohort-specific U-Nets and a shared classifier reached 0.67
against 0.41 for the original fine-tuned on the same folds. On an external public SWI cohort (57 cases) the VALDO-only chain scored
0.61-0.65 without any adaptation (median per subject 0.67). About forty further interventions -- loss functions, augmentation,
intensity harmonisation, synthetic lesions, longer training, second-stage variants, shape filters against vessels -- raised sensitivity
but not F1: the remaining bottleneck is separating microbleeds from vessel cross-sections with a single magnitude sequence, most of all
under heavy loads of small, low-contrast lesions. The resulting chain is trainable from public data alone and is released with code,
commands and all negative results.

## 1 Ausgangspunkt: MicrobleedNet auf VALDO
Zweistufig (CDet 48-mm-Würfel, CDisc 24-mm-Würfel, Lehrer-Schüler), Gefäß-Übermalung als Vorverarbeitung, FRST als
Zusatzkanal, invertierte Skala 1 - I/max. Nachbau mit Attention-U-Net und Originalrezept auf VALDO-T2*: **F1 0.245**
(Precision 0.20). Drei Fehler im Originalcode dokumentiert (ABWEICHUNGEN-microbleednet.md A1-A3).

## 2 Die sieben Änderungen, die blieben (jede mit der Messung, die sie trägt)
| Nr. | Baustein | Original | Unsere Kette | Messung | Warum |
|---|---|---|---|---|---|
| 1 | Gefäß-Übermalung | ja (vor dem Training) | **keine** | **+0.136 gesichert** (4.1) | Übermalung traf 109 von 236 Blutungen, 17 wurden unsichtbar; Gefäße erkennt das Netz selbst |
| 2 | Architektur | Attention-U-Net-artig (CDet) | **anisotropes 3D-U-Net a03-aniso** (erste Stufe nur in der Schicht) | **+0.159 gesichert** im Turnier über 9 Netze; auf sauberen Daten bestätigt (nnU-Net 0.458, Attention 0.463, a03 0.585) | dicke Schichten (0.8-5 mm) verlangen anisotrope Faltung |
| 3 | Gitter | Original-Auflösung | **0.5 x 0.5 x 1 mm**, Hirnzuschnitt | z = 1 gegen 2 mm +0.036 (4.8) | feine z-Abtastung trennt "endet" von "läuft weiter" |
| 4 | FRST-Kanal | ja | **behalten** | +0.053 (4.11) | einziger Vorverarbeitungshebel mit gesichertem Gewinn |
| 5 | Stichprobe | 1:x Klassen-Verhältnis | **60 % Läsion / 25 % Hirn / 15 % beliebig**, 800 je Epoche, 60 Epochen, Zwischenstand auf innerer Falte | 1:2 kollabiert (5g); 120 Epochen -0.022 gesichert (5h) | Budget je Fall ist der Engpass, nicht Epochen |
| 6 | Nachbearbeitung | Schwelle | **feste Zelle 0.3 / 2 mm3 + Liquor-Regel** (SynthSeg: Mehrheit im Liquor -> weg) | **+0.06, 0 Blutungen verloren** (4.3) | 98 von 176 Fehlalarmen lagen im freien Liquor |
| 7 | Zweite Stufe | CDisc je Datensatz, Lehrer-Schüler | **ein gemeinsamer 3D-CNN-Klassifikator über alle Kohorten** (16-mm-Würfel, Bild + FRST + Wahrscheinlichkeit), eigenes U-Net je Kohorte | SWI **+0.045 bestätigt / +0.058 (10 Startwerte)**; VALDO / T2* gleich der Liquor-Regel; Lehrer-Schüler +0.003 (5z) | Pool aller Kohorten ist groß genug, je Kohorte war er es nicht; ersetzt auf SWI die harte Regel, die dort 74 echte Blutungen löscht |

Ergebnis der Kette auf VALDO: **0.245 -> 0.647** (Zwei-Startwerte-Ensemble mit Liquor-Regel; Mittel über Einzelstartwerte 0.605 ± 0.034).

## 3 Übertragung auf die eigene Klinik-Kohorte (der eigentliche Zweck)
| Strategie | catalina-T2* (75 Fälle, 828 CMB) | catalina-SWI (61, 652) |
|---|---|---|
| VALDO-Modell ohne Anpassung | 0.622 | 0.305 |
| gemischt / vortrainiert + feingetunt | -0.02 bis -0.045 gesichert | -0.045 gesichert |
| **eigenes Training je Kohorte, 60 Epochen** | **0.663** | **0.626** |
| + gemeinsamer Klassifikator | **0.671** | **0.672** |
Befund: U-Nets je Kohorte, Klassifikator gemeinsam. Und der Klassifikator, NUR auf VALDO trainiert, ist ohne Feintuning
gleich gut (SWI 0.671, T2* 0.648) -> eine rein öffentliche, veröffentlichbare Kette.

## 4 Ehrliche Prüfung außerhalb der Kreuzvalidierung
| Prüfung | Ergebnis |
|---|---|
| 15 beiseitegelegte VALDO-Fälle, einmalig, eingefrorene Kette | gepoolt 0.468; ein Fall trägt 73 der 97 Blutungen (Sensitivität dort 0.20); ohne ihn 0.731; Median je Fall 0.667 |
| Momeni/CSIRO, fremde öffentliche SWI-Kohorte (57 Fälle, 146 Blutungen), ohne Anpassung | VALDO-U-Net + Liquor-Regel 0.613, + Laufstreckenregel 0.645; SWI-U-Net + VALDO-Klassifikator 0.630; Median je Proband 0.667; Sensitivität ~0.8 ohne Regeln |
Grenze der Kette: massive Last kleiner, kontrastarmer Blutungen (Sensitivität 0.20 auf dem schwersten Fall).

## 5 Was nicht half (eine Tabelle im Anhang, hier nur der Satz)
Kein Trainingshebel (blob loss, Zentrumskarte, fnrw, harte Negative, Augmentierung, 120 Epochen, Ausschnittgröße, künstliche
Blutungen in zwei Fassungen), keine Intensitätsangleichung (z, Landmarken, p99.5, kubisch), keine Variante der zweiten Stufe
(zweiskalig, Lehrer-Schüler, Vortraining, mehr Kandidaten) hob das F1. Das Muster war immer dasselbe: mehr Empfindlichkeit,
mehr Fehlalarme. Der Engpass ist die Trennung Blutung / Gefäß mit einer Sequenz -- und Formfilter (Laufstrecke, Frangi)
scheitern auf dem anisotropen Gitter, weil Blooming eine Blutung zum kurzen Zylinder macht.

## 6 Vorgeschlagene Gliederung (kurz)
1 Einleitung (CMB, MicrobleedNet als Ausgangspunkt, Ziel: eigene Klinik-Kohorte) · 2 Daten (VALDO 3 Kohorten, catalina T2*/SWI,
Momeni) und Metrik · 3 Nachbau und Fehler des Originals · 4 Die sieben Änderungen (Tabelle 2, je ein Absatz mit Abbildung
aus dem Schichtbetrachter) · 5 Übertragung (Tabelle 3) · 6 Prüfung außerhalb der CV (Tabelle 4) · 7 Grenzen und Negativbefunde
(ein Absatz + Anhangstabelle) · 8 Schluss: veröffentlichbare Kette, Grenze bei massiver Last.
Abbildungen: (a) Übermalung löscht Blutungen (4.1), (b) Kettenschema, (c) Beispielfälle aus dem Betrachter (Referenz gegen
Basis+Liquor), (d) Balken der sieben Änderungen, (e) Momeni-Beispiel. Quellen für alle Zahlen: ARBEIT.md, protokoll.md,
REPLIKATION.md (wörtliche Befehle), UEBERSICHT-tests-0929.md (alle Tests).

## 7 Welche Zahlen die Arbeit braucht -- und welche nicht (Nutzerfrage 29.09.)

**Hauptmaß (eine Zahl je Vergleich):** Läsions-F1, gepoolt über alle Fälle (26er-Nachbarschaft, Berührungsregel), an der festen
Zelle 0,3 / 2 mm3, IMMER gepaart gegen die Basis mit Bootstrap über Fälle (10 000 Züge, 95-%-Intervall). Ohne Intervall ist keine
Zahl in der Arbeit deutbar; ohne zweiten Startwert ist beim U-Net nichts unter 0,08 deutbar (Abschnitt 3 in ARBEIT.md).
**Nebenmaße (immer zusammen berichten, nie einzeln):** Sensitivität, Precision, Fehlalarme je Fall (das Muster "Sensitivität hoch,
F1 runter" ist die wichtigste Erkenntnis der Arbeit und nur an diesen dreien sichtbar), dazu Median-F1 je Proband nach
Challenge-Regel (Vergleichbarkeit mit VALDO-Paper und CenSynCMB) und die Sensitivität je Größenklasse (< 7 / 7-14 / > 14 mm3).
**AUC:** nur dort, wo eine STETIGE Größe Kandidaten trennt -- Kandidatenmerkmale (Abb. ROC) und die Ausgabe des Klassifikators.
Eine Voxel-AUC des Detektors ist bei 0,01 % positiven Voxeln bedeutungslos und wird nicht berichtet.
**Loss:** kein Ergebnismaß. Eine Abbildung: Trainingsverlust 60 / 90 / 120 Epochen neben dem F1 -- der Verlust fällt weiter, das F1
nicht (Budget je Fall, 5h). Das ist das Argument gegen "länger trainieren".
**Volumen (ml):** beschreibend (Referenz gegen Vorhersage je Kohorte, eine Tabelle), nicht als Zielgröße -- CMB-Volumen hängt am
Blooming, klinisch zählt die ZAHL. Stattdessen: Zählfehler je Fall (|vorhergesagt - Referenz|) und Fall-Genauigkeit (hat der Fall
überhaupt Blutungen), beide liegen in `evaluate`.
**Streuung:** Startwert-Rauschen U-Net (drei Startwerte: 0,646 / 0,590 / 0,578) und zweite Stufe (10 Startwerte: sd 0,003-0,009)
je einmal beziffern; danach jede Entscheidung mit zwei Startwerten.

**Protokoll "eine Änderung gegen die Basis":** ja, genau so -- und in der Arbeit als KUMULATIVE Kette erzählen (Tabelle 2), nicht als
40 Einzelvergleiche: 0,245 (Nachbau) -> 0,404 (ohne Übermalung + anisotropes Netz, Turnier) -> 0,487 (+ Liquor-Regel) -> 0,585 roh /
0,646 mit Regel (Gitter d, 60 Ep., FRST, Stichprobe) -> 0,647 (zwei Startwerte). Jeder Schritt hat seine gepaarte Zahl gegen den
Vorgänger. Die Regeln, die das Verfahren ehrlich machen: Erwartung und Entscheidungsregel VOR dem Lauf aufschreiben; Prüfmenge nur
einmal berühren; nichts nachträglich umdeuten; negative Ergebnisse alle nennen (Anhang). Was NICHT gemacht wurde und im Ausblick
steht: eine Rückwärts-Ablation der Endkette (jede Änderung einzeln wieder herausnehmen) -- teils vorhanden (ohne FRST -0,053,
ohne Liquor-Regel -0,06, Attention statt aniso -0,12), teils nicht (Übermalung wieder hinein).
