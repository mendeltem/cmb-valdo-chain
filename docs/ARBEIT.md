# Mikroblutungen auf T2*: das Original unter der Lupe und eine begruendete Verbesserung

Arbeitsfassung im Stil einer Masterarbeit (Nutzerwunsch 19.09.2026). Regel fuer jeden Eingriff:
**Original -> Beobachtung (gemessen) -> Hypothese -> Aenderung -> Experiment -> Ergebnis -> Entscheidung -> Grenzen.**
Keine Aenderung ohne Messung davor und danach; "gemessen" und "interpretiert" stehen getrennt; negative Ergebnisse
bleiben drin. Zahlen stammen aus `protokoll.md` (dort mit Datum und Datei), die Abweichungstabelle aus
`ABWEICHUNGEN-microbleednet.md`. Private Kohortendaten (catalina) erscheinen nur als Summen.

## 1 Fragestellung
1. Wo verliert das veroeffentlichte MicrobleedNet-Verfahren (Sundaresan et al.; Code Commit 958f1cb) auf heterogenen
   T2*-GRE-Daten (VALDO Task 2: drei Kohorten, 0.8 bis 4 mm Schichtdicke) Leistung -- in der Vorverarbeitung, im Netz,
   im Training oder in der Nachbearbeitung?
2. Welche Aenderungen beheben das nachweisbar, gemessen am Laesions-F1 in geschichteter 5-fach-Kreuzvalidierung?
3. Haelt der Gewinn auf einer unabhaengigen Kohorte mit anderem Scanner (catalina-T2*, 75 Faelle, 830 CMB)?

## 2 Material
VALDO Task 2: 72 Faelle, 236 CMB; Kohorte 1 0.45x0.45x4 mm (11 Faelle), Kohorte 2 0.49x0.49x0.8 mm (34), Kohorte 3
1x1x3-4 mm (27). Aufteilung: 15 Faelle (97 CMB) beiseite, 57 Faelle (139 CMB) in 5 geschichteten Falten; innere Falte
waehlt Epoche/Schwelle. catalina: T2* 75/830, SWI 61/653; Kontrast der CMB 0.30 (VALDO 0.46-0.51), Schicht 5 mm.

## 2a Stand der Technik: zwei Linien, die einander nicht kennen
MicrobleedNet (Preprint 11/2021, Paper 2023) und die VALDO-Challenge (MICCAI 2021, Paper 2024) entstanden parallel.
Das MicrobleedNet-Paper nutzt 20 VALDO-Faelle nur zum Einstellen von Hyperparametern und vergleicht sich mit keinem
VALDO-Team; umgekehrt verwendete kein VALDO-Team FRST, Gefaessentfernung oder Destillation. Die VALDO-Spitze gewann mit
langem Training, grossem 3D-Kontext und drei Modalitaeten; MicrobleedNet setzt auf Vorwissen (FRST, Gefaessmaske,
Formfilter, Randabstand) und eine zweite Stufe. Diese Arbeit prueft beide Ideenfamilien auf denselben Daten.
Einzelheiten und Quellen: `recherche/sundaresan-gegen-valdo.md`, `recherche/valdo-challenge.md`.
Das Original-Paper nennt zwei Grenzen selbst, die hier gemessen werden: Gefaessentfernung kann CMB entfernen
(bei uns 109 von 236), Drehung/Skalierung kann CMB zerstoeren (unser negatives Augmentierungs-Ergebnis).

## 3 Methode der Untersuchung
- Masse: gepooltes Laesions-F1 (26er-Komponenten, Beruehrung) als Rangmass; Precision, Sensitivitaet, FP je Fall;
  zusaetzlich die offizielle VALDO-Regel (Median-F1 je Proband, 5 mm) zur Einordnung gegen die Challenge.
- Vergleiche immer GEPAART ueber dieselben Faelle (Bootstrap ueber Faelle, 95-%-Intervall).
- Rauschmass: gleicher Lauf mit anderem Startwert. Im 40-Epochen-Rezept mit Fruehstopp rund 0.08 F1 -- deshalb seit
  19.09. feste 60 Epochen fuer alle Vergleiche. **Praezisierung 21.09. (Begriffspruefung, Abschnitt 11):** "fest" heisst: der Lernratenplan laeuft immer 60 Epochen, es gibt keinen
  Fruehstopp -- die Testvorhersage kommt aber vom Zwischenstand mit dem besten F1 der INNEREN Falte (Auswertung alle 5 Epochen; `cv5.py`: `model.load_state_dict(best["state"])`), nicht von Epoche 60.
  Nachgemessen im neuen Rezept (60 Ep. fest, ohne Uebermalung): Startwert 1 gegen 42 = 0.585 gegen 0.588, gepaart
  -0.003 [-0.066; +0.059]; je Falte hoechstens 0.06 Abstand. Das Rezept ist stabil; als gesichert gilt, was das
  gepaarte Intervall von rund +-0.06 verlaesst.
  **Nachtrag 22.09. (dritter Startwert, 5u): das Rauschmass war zu klein angesetzt.** Drei Startwerte des identischen Rezepts auf gitter_d: 0.585 / 0.553 / 0.501 roh, 0.646 / 0.590 / 0.578 mit Liquor-Regel --
  Spanne 0.07-0.08, und Startwert 2 ist gegen Startwert 42 gepaart "gesichert" schlechter (-0.083 [-0.135; -0.033]). Der Bootstrap ueber Faelle misst also NUR die Unsicherheit der Fallauswahl bei EINEM
  trainierten Netz, nicht die des Trainings. Folgen fuer alle Vergleiche in diesem Dokument: (1) ein Effekt aus EINEM Startwert je Seite gilt erst ab ~0.08 als belastbar, darunter braucht er einen zweiten
  Startwert oder eine unabhaengige erste Stufe (so gehandhabt seit 5j); (2) "gesicherte" Einzelbefunde unter 0.08 (z. B. letzter Stand s42 -0.047, catalina 120 Ep. -0.022, blob -0.044) sind Richtungsangaben,
  keine Beweise -- genau deshalb wurde blob mit zweitem Startwert bestaetigt und der letzte Stand mit Startwert 1 NICHT; (3) Effekte ueber 0.08 (Uebermalung +0.136, 24-mm-Ausschnitt -0.216, kein FRST
  -0.053 knapp darunter) bleiben belastbar. Fuer das Paper: Basis als Mittel +- Spanne ueber drei Startwerte berichten (0.605 +- 0.034 mit Regel), nicht 0.646.
- Gittersuchen nur mit kreuzweiser Zellenwahl; unter gleichwertigen Einstellungen gewinnt die einfachste.

## 4 Befunde und Eingriffe (jeder als Argumentkette)

### 4.0 Lauffaehigkeit des veroeffentlichten Codes (Fork `custom_microbleednet`, unser Beitrag)
- **Original:** Commit 958f1cb. **Beobachtung (am Quelltext gelesen, noch nicht ausgefuehrt):** drei Aufrufe im
  Feintuning-Pfad passen nicht zu den Signaturen -- `commands.py` uebergibt `aug/save_cp/save_wei` an ein `main()`, das
  `perform_augmentation/save_checkpoint/save_weights` erwartet; `CDetPatchDataset` wird ohne das Pflichtargument `ratio`
  gebaut; `CDiscPatchDataset` bekommt die Klassen vertauscht (negativ, positiv statt minority, majority) und ebenfalls
  kein `ratio`. Ausserdem friert `freeze_layers_for_finetuning` den Klassifikator-Kopf von CDisc immer ein.
- **Folge (beobachtet in der MB-Arena):** CDisc kollabiert beim Feintuning auf "alles negativ" (Genauigkeit 0.5).
- **Aenderung im Fork:** Aufrufe repariert, Verhaeltnis positiv:negativ auf 1:1 festgelegt, Kopf (und `inpconv`)
  aufgetaut, Hirnmaske mit gefuellten Loechern, Kandidatenschwelle 0.2 -> 0.1 (auf catalina-SWI lieferte 0.2 in 59 von
  61 Faellen keine Kandidaten). Einzeln mit Staerke: `ABWEICHUNGEN-microbleednet.md`, A1-A9.
- **Grenze:** A1-A3 sind gelesen, nicht ausgefuehrt; vor einer Aussage nach aussen am frischen Klon nachstellen.

### 4.1 Gefaess-Uebermalung
- **Original:** `inpaint_vessels`: Frangi je Schicht (`beta=20`, Skalen in Pixeln), KMeans k=2 je Schicht, gerettet wird,
  was rund und solide ist.
- **Beobachtung:** trifft 109 von 236 CMB, 27 ganz, 17 danach unsichtbar; uebermalt 12-17 % des Hirns; Schaden fast nur
  bei 3-4-mm-Schichten (Kohorte 2: 0 unsichtbar). FRST-Treffer an der CMB 85 % mit, 96 % ohne Uebermalung (nachgerechnet 21.09. mit `cmb/analysis/frst_hit_rate.py`: 82 % / 96 %, siehe Abschnitt 10).
- **Hypothese:** `beta=20` schaltet die Unterdrueckung runder Flecken ab; Pixel-Skalen und der erzwungene Cluster je
  Schicht machen das Verfahren von Aufloesung und Schichtinhalt abhaengig.
- **Aenderung:** Uebermalung 2 (`code/uebermalung2.py`): beta 0.5, Skalen in mm, eine Schwelle je Volumen, nur lange
  duenne Komponenten. Schaden 1 von 236 bei 0.9-1.5 % uebermaltem Hirn; traegt unveraendert auf catalina-T2* (1/830)
  und -SWI (11/653 beruehrt, 0 zur Haelfte).
- **Experiment (laeuft):** Dosis-Wirkungs-Reihe keine / 2 %,12 mm / 3 %,8 mm / 5 %,6 mm / Original, je 60 Epochen;
  dazu Gittersuche als Nachfilter (50 Zellen, ohne Training).
- **Ergebnis Nachfilter:** aendert F1 in keiner Zelle (0.5236): 0 von 37 Vorhersagen liegen auf langen Gefaessen.
- **Ergebnis Training OHNE Uebermalung (gitter_d, 60 Ep.; Stand 15:30 mit 4 von 5 Falten, Endstand 5/5 darunter):** F1 0.591 gegen 0.455 der
  Basis auf denselben 46 Faellen, gepaart **+0.136 [+0.065; +0.226]**, gesichert. Sensitivitaet 0.42 -> 0.57, Precision
  0.50 -> 0.62, Fehlalarme je Fall 1.0 -> 0.9; je Falte 0.65 0.50 0.63 0.56 gegen 0.48 0.35 0.48 0.48. Die
  Original-Uebermalung kostet also rund 0.14 F1 -- und das Weglassen bringt NICHT mehr Fehlalarme, sondern weniger.
- **Endstand 5/5 Falten (16:00):** F1 0.588 (P 0.60, S 0.58, 0.93 FP je Fall); mit Liquor-Regel 0.612 (P 0.66, S 0.57, 0.70).
  Nach Challenge-Regel (Median-F1 je Proband, 5 mm): 0.667 [0.11; 0.76] -- dieselbe Hoehe wie die Challenge-Plaetze 1-3
  (0.667-0.684), allerdings CV auf 57 Faellen gegen verdeckten Test mit 147, also kein Rangvergleich.
- **Uebermalung 2 (3 %, 8 mm; gitter_e) gegen KEINE, 60 Ep., 5/5 Falten (19.09. 18:00):** F1 0.584 gegen 0.588, gepaart
  -0.005 [-0.065; +0.058] -- kein Unterschied. Sie verschiebt nur das Gleichgewicht (Precision 0.60 -> 0.64,
  Sensitivitaet 0.58 -> 0.54, Fehlalarme je Fall 0.93 -> 0.75). Vorsichtige Stufe 2 %/12 mm (gitter_f): F1 0.588 gegen 0.588, gepaart -0.000 [-0.063; +0.055]. Kraeftige Stufe 5 %/6 mm (gitter_g): 0.570, gepaart -0.018 [-0.074; +0.028].
  **Dosis-Wirkung komplett:** keine 0.588 | 2 %/12 mm 0.588 | 3 %/8 mm 0.584 | 5 %/6 mm 0.570 | Original 0.470 -- je mehr uebermalt wird, desto schlechter; nirgends ein Gewinn.
- **Entscheidung:** Original-Uebermalung streichen. Ob Uebermalung 2 (gitter_e/f/g) gegenueber "keine" noch
  etwas bringt, zeigen die naechsten drei Laeufe; nach dem Nachfilter-Ergebnis ist wenig zu erwarten.

### 4.2 Hirnmaske
- **Original / Ausgangslage:** Maske = Bild > 0; VALDO liefert "masked".
- **Beobachtung:** nur Kohorte 1 ist skull-stripped (1419 ml); Kohorte 2 1770 ml (16 % Nicht-Hirn), Kohorte 3 2533 ml
  (40 %: Knochen, Kopfhaut, Schaedelbasis). Keine CMB wird abgeschnitten.
- **Hypothese:** dunkler Knochen neben dem Hirn erzeugt Verwechslungen und verfaelscht die Intensitaetsskala.
- **Aenderung:** SynthSeg-Hirn + 2 Voxel (`--maske synthseg`), entfernt 19 % der alten Maske, 0 von 139 CMB ausserhalb.
- **Ergebnis (60 Ep., gemessen 19.09.):** F1 0.421 gegen 0.470 der Basis, gepaart -0.048 [-0.099; +0.009]; je Falte
  0.47 0.32 0.36 0.44 0.55 gegen 0.48 0.35 0.48 0.48 0.53. Kein Gewinn, eher Verlust (nicht gesichert).
- **Entscheidung:** enge Maske als Eingangsmaske NICHT uebernehmen. **Interpretiert / offen:** die Maske wurde nur um
  2 Voxel (1 mm in der Schicht) geweitet -- moeglicherweise fehlt dem Netz der Kontext an der Hirnoberflaeche, wo viele
  lobaere CMB liegen; ein Rand von 5 mm waere der faire zweite Versuch. Als NACHFILTER bleibt die Anatomie wirksam
  (Liquor-Regel). **Lehre:** "maskiert" zweifach pruefen -- schneidet sie ab UND wie viel Nicht-Hirn bleibt; und: eine
  plausible Reparatur ist noch kein Gewinn, erst die Messung entscheidet.

### 4.3 Fehlalarme im Liquor
- **Beobachtung:** 98 von 176 Fehlalarmen liegen im freien Sulcus-Liquor (SynthSeg-Label 24), nicht in den Ventrikeln.
- **Aenderung:** parameterfreie Regel "Komponente mit Mehrheitslabel Liquor verwerfen".
- **Ergebnis:** F1 0.245 -> 0.335 (Attention-U-Net), 0.404 -> 0.487 (a03, 40 Ep.), 0.470 -> 0.524 (a03, 60 Ep.); TP-Verlust 0-2.
- **Nebenbefund:** ein frueherer Filter (we5) mass das Gegenteil, weil seine Labeltabelle falsch war
  (linkes Putamen/Pallidum statt Ventrikel). **Grenzen:** Regel nach Blick auf die CV-Daten formuliert; 5 von 139
  Referenz-CMB liegen selbst im Liquor.

### 4.4 Architektur
- **Original:** zweistufig (Kandidaten 48^3 + Klassifikator 24^3, Destillation).
- **Experiment:** Turnier ueber neun einstufige 3D/2.5D-Netze im selben Ablauf.
- **Ergebnis:** einzig gesicherter Gewinn: anisotropes U-Net (erste Stufe nur in der Schicht) +0.159 [+0.091; +0.234]
  gegen das Attention-U-Net; gewinnt in allen drei Kohorten. Tiefe/grosse Netze (Residual-SE, SegResNet, Swin)
  gesichert schlechter bzw. nicht trainierbar im gemeinsamen Rezept. Attention-Gates auf dem Sieger: kein erkennbarer
  Gewinn (40 Ep.; 60 Ep. eingereiht).
- **Interpretiert:** bei interpolierten dicken Schichten hilft es, zuerst nur der gemessenen Richtung zu trauen.
- **Erledigt (19.-21.09.):** die zweistufige Fassung auf dem Sieger ist gebaut und gemessen -- 5d (je Kohorte, ohne Gewinn) und 5j (gemeinsam ueber alle Kohorten, auf SWI gesichert besser).

### 4.5 Trainingsrezept
- **Beobachtung:** zweiter Startwert 0.321 statt 0.404; Ursache eine Falte mit Fruehstopp bei Epoche 5.
- **Aenderung:** feste 60 Epochen. **Ergebnis:** 0.404 -> 0.470 roh, alle Falten 0.35-0.53. 300 Epochen eingereiht.
- **Literaturstuetze:** die drei besten VALDO-Teams trainierten lang (nnU-Net 1000 Epochen), mit grossem Kontext und
  T1+T2+T2* (Sudre et al. 2024, arXiv 2208.07167; von mir nicht zeilenweise gegengelesen).

### 4.7 Intensitaetsnormierung + Augmentierung (negatives Ergebnis, 19.09.)
- **Beobachtung:** Skala zwischen Kohorten nicht harmonisiert (Eingang = 1 - Bild/Maximum; 5.-95. Perzentil im Hirn
  0.25-0.74 in Kohorte 1 gegen 0.65-0.98 in Kohorte 2); Augmentierung bisher nur Spiegelungen.
- **Hypothese:** z-Normierung je Fall und reichere Augmentierung (Drehung +-15 Grad, Massstab, Kontrast, Rauschen)
  verbessern die Verallgemeinerung.
- **Experiment:** a03-aniso, 60 feste Epochen, beides ZUSAMMEN als Paket (`--norm z --aug`), gepaart gegen die Basis.
- **Ergebnis:** F1 0.410 gegen 0.470 (mit Liquor-Regel 0.472 gegen 0.524); gepaart -0.060 [-0.135; +0.011]; alle fuenf
  Falten niedriger (0.43 0.33 0.40 0.43 0.45 gegen 0.48 0.35 0.48 0.48 0.53). Nicht gesichert, aber einheitlich.
- **Entscheidung:** Paket nicht uebernehmen. **Grenzen / offen:** (a) das Paket trennt nicht, ob Normierung oder
  Augmentierung schadet; (b) die z-Normierung rechnet ueber die VALDO-Maske, die in Kohorte 3 zu 40 % Nicht-Hirn ist --
  ein fairer Test braucht die enge Maske (`--maske synthseg --norm z`); (c) Augmentierung braucht ueblicherweise MEHR
  Epochen, 60 koennen zu wenig sein; die Drehung interpoliert die wenigen Voxel einer CMB (das VALDO-Paper warnt davor).

### 4.8 Umrechnung auf das gemeinsame Gitter (Befund 19.09., Experimente laufen)
- **Ausgangslage:** alle Faelle werden auf 0.5 x 0.5 x 1.0 mm umgerechnet (Bild linear, Label naechster Nachbar).
- **Beobachtung (gemessen, 57 CV-Faelle, Daten ohne Uebermalung):** das Label-Volumen bleibt erhalten (-1 / 0 / 0 %),
  aber der Kontrast der CMB gegen einen Ring von 0.5-2 mm in der Schicht sinkt: Kohorte 1 0.148 -> 0.114 (-23 %),
  Kohorte 2 0.123 -> 0.109 (-12 %), Kohorte 3 0.128 -> 0.087 (-32 %) [nachgerechnet 21.09. mit `cmb/analysis/resampling_contrast.py`: -26 / -15 / -30 %, siehe Abschnitt 10]. (Ein erster Versuch mit einer Schale in
  Voxeln statt mm ergab -36 / -14 / -36 %, war aber zwischen den Gittern nicht vergleichbar und ist verworfen.)
- **Hypothese:** lineare Interpolation verschmiert Objekte von wenigen Voxeln; am staerksten dort, wo am meisten
  hochgerechnet wird (Kohorte 3: 1 mm -> 0.5 mm in der Schicht und 3-4 mm -> 1 mm in z; Median der CMB dort 4 Voxel).
- **Experimente:** (a) Gitter mit z = 2 mm statt 1 mm (`dev/gitter_z2`, eingereiht); (b) kubische statt linearer
  Interpolation des Bildes (`GITTER_C_ORDNUNG=3`, Kontrast erst auf der CPU nachmessen, dann trainieren).
- **Ergebnis z = 2 mm (20.09. 02:40, 5/5 Falten, Patch 20x62x94):** F1 0.552 gegen 0.588 (z = 1 mm); gepaart -0.036 [-0.105; +0.032].
- **Entscheidung nach der vorab festgelegten Regel: es bleibt bei z = 1 mm.** Weniger Interpolation hilft nicht; vermutlich braucht
  das Netz die feinere z-Abtastung, um "endet nach einer Schicht" von "laeuft weiter" zu unterscheiden. Einschraenkung: andere
  Gittergroesse heisst auch andere Label-Rasterung, der Vergleich ist nicht voxelgleich. Kubische Interpolation bleibt offen.

### 4.9 Bias-Korrektur und Intensitaetsskala (Experimente eingereiht)
- Bias-Korrektur (FSL FAST -B, bei uns vor der Kette): wirkt laut Statistik in Kohorte 1 deutlich, in Kohorte 2 kaum;
  ihr Nutzen fuers F1 wurde nie gemessen -> `dev/gitter_h` ohne Bias-Korrektur, gepaart gegen gitter_d.
  Endstand 20.09. (5/5 Falten): ohne Bias-Korrektur 0.566 gegen 0.588, gepaart -0.022 [-0.097; +0.052] -- kleiner, nicht gesicherter
  Nachteil; die Korrektur bleibt (sie schadet nicht, und fuer catalina liegt sie schon vor).
  Perzentil-Skala p99.5 (5/5 Falten): 0.613 gegen 0.588, gepaart +0.025 [-0.039; +0.092]; vier von fuenf Falten besser
  (0.67 0.49 0.61 0.65 0.65 gegen 0.65 0.50 0.63 0.56 0.58). Nicht gesichert, aber die einzige Variante mit positivem Vorzeichen
  -> Kandidat fuers Endrezept, mit zweitem Startwert zu bestaetigen. (Im verworfenen Paket 4.7 steckte die z-Normierung, nicht diese Skala.)
- Skala: Original teilt durch das Maximum (ausreisserempfindlich) -> `--norm p995` teilt durch das 99.5-Perzentil im
  Hirn; getrennt von der Augmentierung, die im Paket 4.7 mitgemessen wurde.

### 4.10 Zusatzkanal: die Zielkohorte entscheidet
catalina hat KEIN T1. FLAIR liegt lokal nur fuer 17 von 75 T2*-Faellen (23 von 61 SWI) vor, 0.7 x 0.7 x 2.5 mm, nicht im
T2*-Raum (gezaehlt 19.09.). Der T1-Kanal der VALDO-Spitze ist damit fuer das Transferziel wertlos. Pruefbar auf VALDO
ist nur T2 als Stellvertreter fuer FLAIR; das braucht echte Mehrkanal-Unterstuetzung und kommt mit dem Paket `cmb/`.
Bis dahin bleibt das Transfermodell bei T2* + FRST -- beides laesst sich auf jeder Kohorte aus dem T2* allein rechnen.
Vom Nutzer bestaetigt (19.09.): in der catalina-Datenbank hat NUR eine Untergruppe FLAIR, der Rest nur SWI oder T2*. Ein Modell,
das FLAIR als Pflichteingang braucht, waere auf dem Grossteil der Kohorte nicht anwendbar -> das Detektionsmodell bleibt
einsequenzig (T2* bzw. SWI + daraus gerechnete FRST). FLAIR kommt hoechstens als OPTIONALER Kanal mit Kanal-Dropout in Frage
(40 Faelle, Nutzen fraglich) oder fuer Nebenanalysen der Untergruppe (z. B. CMB-Last gegen WMH-Last).

### 4.11 Hilft FRST? (Zwischenstand 19.09., entscheidender Lauf vorgezogen)
- **Original:** FRST (radiale Symmetrie) ist im MicrobleedNet ein Pflichtkanal neben dem Bild.
- **Gemessen bisher:** (a) FRST markiert die Blutungen zuverlaessig: bei 96 % liegt das Maximum im obersten 1 % der Hirnvoxel
  (ohne Uebermalung). (b) Sie trennt aber NICHT echte von falschen Kandidaten: AUC 0.56 (0.5 = wertlos); FRST >= 0.9 haben 96 %
  der echten und 88 % der falschen Kandidaten. Zum Vergleich: Netz-Wahrscheinlichkeit und Ausdehnung 0.69-0.70, Kontrast 0.65.
  (c) Im Random Forest der zweiten Stufe traegt sie 0.01 bei. (d) Alt, 40 Epochen, uebermalte Daten: mit FRST 0.215, ohne 0.175,
  nicht gesichert.
- **Interpretiert:** FRST antwortet auf alles Dunkle und Runde -- also auch auf genau die Verwechslungskandidaten. Als Wegweiser
  ("hier hinschauen") kann sie dem Netz trotzdem helfen; als Unterscheidungsmerkmal taugt sie nicht.
- **Experiment:** a03-aniso, 60 Ep., gitter_d, nur T2* (`--kanaele t2s`), gepaart gegen die Basis mit FRST; laeuft als Naechstes.
- **Ergebnis (20.09. 02:40, 5/5 Falten):** ohne FRST F1 0.535 (P 0.52, S 0.55, 1.21 FP je Fall) gegen 0.588 mit FRST; gepaart
  -0.053 [-0.116; +0.004]; in allen fuenf Falten niedriger (0.63 0.44 0.54 0.54 0.52 gegen 0.65 0.50 0.63 0.56 0.58).
- **Entscheidung nach der vorab festgelegten Regel (einfachere Variante nur bei Delta >= -0.02): FRST BLEIBT.**
- **Interpretiert:** FRST trennt echte von falschen Kandidaten nicht (AUC 0.56), hilft dem Netz aber als Wegweiser -- ohne sie
  sinken Sensitivitaet UND Precision. Das stuetzt den Kanal des Originals; fuer jede neue Kohorte muss FRST also gerechnet werden
  (aus dem T2*/SWI allein, kein Zusatzbild noetig).

### 4.6 Geprueft und in Ordnung
Bias-Korrektur 72/72 gelaufen; keine CMB ausserhalb Maske/Zuschnitt; FRST-Lage ohne Versatz (0.1 Voxel; ein scheinbarer
Versatz von 0.9 Voxel war ein Messartefakt durch argmax auf einem Plateau).

## 5 Stand der Leistung (VALDO-CV, 57 Faelle, out-of-fold)
| Stufe | F1 roh | F1 mit Liquor-Regel | Median-F1 je Proband (VALDO-Regel) |
|---|---|---|---|
| Ausgang: 3D-Attention-U-Net, 40 Ep. | 0.245 | 0.335 | 0.21 / 0.32 |
| anisotropes U-Net, 40 Ep. | 0.404 | 0.487 | 0.48 / 0.56 |
| anisotropes U-Net, 60 Ep. fest | 0.470 | 0.524 | noch zu rechnen |
| ... + z-Normierung + Augmentierung | 0.410 | 0.472 | -- (verworfen) |
| ... + enge SynthSeg-Maske | 0.421 | -- | -- (verworfen) |
| ... OHNE Gefaess-Uebermalung (5/5 Falten) | 0.588 | 0.612 | 0.667 / 0.667 |
| ... + gemeinsame Schwelle und Mindestgroesse, kreuzweise gewaehlt | -- | 0.633 | noch zu rechnen |
| SHIVA-CMB ohne Anpassung (kennt 60 von 72 VALDO-Faellen aus dem Training!) | 0.575 | -- | 0.667 |
| Challenge-Spitze (anderer Testsatz) | -- | -- | 0.67-0.68 |

### 5c Fehleranalyse des besten Modells (a03-aniso, 60 Ep., ohne Uebermalung; 19.09. abends)
Gemessen an den 201 Kandidaten der Pruefseite (Naeherung: "gefunden" zaehlt Vorhersage-Komponenten, nicht Referenzen):
- Von 38 Faellen mit CMB wird in 10 KEINE einzige gefunden; 7 davon sind Kohorte 3, fast alle mit nur 1-2 CMB -- das ist das
  untere Quartil 0 der Challenge-Regel.
- Verpasste CMB (59): die hoechste Netz-Wahrscheinlichkeit IN der Laesion ist im Median 0.12; bei 37 % liegt sie >= 0.2
  (das Netz "ahnt" sie, die Schwelle schneidet sie weg), bei 39 % unter 0.05 (das Netz ist blind).
- Nach Region: lobaer am schwaechsten (rund die Haelfte gefunden), tief und infratentoriell rund drei Viertel, mesiotemporal 0 von 3.
- Fehlalarme (53): 29 sind winzig (< 5 mm3), 13 liegen im Liquor, 31 stammen aus Kohorte 2.
**Gittersuche Schwelle x Mindestgroesse auf den gespeicherten Wahrscheinlichkeitskarten (mit Liquor-Regel, ohne Training):**
Ausgang (Schwelle der inneren Falte, keine Mindestgroesse) F1 0.613; beste Zelle Schwelle 0.3 / 2 mm3: 0.646 (P 0.63, S 0.66);
**kreuzweise gewaehlt (Zelle auf vier Falten bestimmt, auf der fuenften gemessen): 0.633** (P 0.63, S 0.64, 0.93 FP je Fall).
Interpretiert: die innere Falte (~11 Faelle) waehlt zu hohe Schwellen (0.5-0.9); eine EINE gemeinsame, niedrigere Schwelle
plus Mindestgroesse in mm3 ist besser. Der groessere Hebel dahinter: Kandidaten bei niedriger Schwelle + zweite Stufe.

### 5d Zweite Stufe (Kandidaten + Klassifikator) -- negatives Ergebnis, 19.09. abends
- **Original / Literatur:** MicrobleedNet und das VALDO-Team Zihao sind zweistufig; die Kaskade von 2026 entfernt so 77 % der FP.
- **Hypothese (aus 5c):** Kandidaten bei niedriger Schwelle holen die "geahnten" Blutungen, ein Klassifikator sortiert die FP aus.
- **Aufbau (`cmb/stage2/`):** out-of-fold-Karten des besten Netzes, Schwelle 0.15, Liquor-Regel, >= 1 mm3 -> 203 Kandidaten
  (103 auf einer Blutung, 100 falsch); sie erreichen 100 von 139 Referenz-Blutungen (Obergrenze Sensitivitaet 0.72, F1 bei
  perfektem Klassifikator 0.84). Geschachtelte Auswertung: Klassifikator fuer Falte k sieht Falte k nie, seine Schwelle wird
  in einer inneren 4-fach-CV der Trainingsfalten bestimmt.
- **Ergebnis (gepooltes F1, 57 Faelle):** alle Kandidaten annehmen 0.590 | nur Schwelle auf die Stufe-1-Wahrscheinlichkeit
  (geschachtelt gewaehlt) **0.627** | logistische Regression auf 10 Merkmalen 0.616 | Random Forest 0.579 | kleines 3D-ResNet auf
  32x32x16-Patches, 3 Startwerte gemittelt 0.599 (Precision 0.69, Sensitivitaet 0.53).
- **Entscheidung:** keine zweite Stufe auf VALDO allein. **Interpretiert:** rund 160 Trainingskandidaten reichen nicht; die
  publizierten Kaskaden hatten Tausende (11 424 in der Arbeit von 2026). Die wichtigsten Merkmale des Waldes sind die
  Stufe-1-Wahrscheinlichkeiten selbst -- das grosse Netz hat den Kontext schon ausgewertet. FRST und Randabstand tragen nichts bei.
  **Offen:** mit catalina (830 Blutungen) waere die Kandidatenzahl fuenfmal so gross; dann erneut pruefen.

### 5a Fremde Messlatte: SHIVA-CMB ohne Anpassung (19.09.)
Frei verfuegbares 3D-ResU-Net (Tsuchida et al. 2024; Gewichte v2, drei Falten, CC BY-NC-SA 4.0), Wrapper
`cmb/baselines/shiva_cmb.py`, 6 s je Fall auf der CPU. Auf unseren 57 CV-Faellen (Schwelle 0.5):
F1 0.575 (P 0.61, S 0.54, 0.83 FP je Fall); Challenge-Regel Median-F1 je Proband 0.667 [0.00; 0.97].
Je Kohorte: 1 (4 mm) F1 0.39, S 0.25 | 2 (0.8 mm) F1 0.61, S 0.69, 1.7 FP je Fall | 3 (3-4 mm) F1 0.62, P 1.00, S 0.44.
**WICHTIG (im SHIVA-Paper gelesen): SHIVA-CMB wurde AUF VALDO trainiert** -- 60 der 72 oeffentlichen Faelle (SABRE 8, RSS 28,
ALFA 24) im Training, 12 im Test. Die 0.575 sind also KEIN unabhaengiger Vergleich: das Modell kennt die meisten dieser
Faelle. Dass unser Netz in echter Kreuzvalidierung (jeder Fall ungesehen) 0.588 erreicht, ist vor diesem Hintergrund stark.
Der faire Vergleich ist catalina-T2*, das weder SHIVA noch unser Netz gesehen hat. **Ergebnis dort (75 Faelle, 830 CMB,
Schwelle 0.5): F1 0.312, Precision 0.92, Sensitivitaet 0.19, 0.17 Fehlalarme je Fall; Median-F1 je Proband 0.42.**
Von 0.575 auf 0.312: das ist der Domaenenabfall eines breit trainierten Modells (450 Scans, sechs Kohorten) auf einer
fremden Kohorte mit schwaecherem Kontrast, kleineren CMB und 5-mm-Schichten -- und die Latte, die unser Transfermodell
ohne Anpassung uebertreffen muss. (Niedrigere Schwelle nicht geprueft.)
Gemeinsame Schwaeche: Kohorte 1 (4-mm-Schichten) -- auch SHIVA findet dort nur ein Viertel der Blutungen.

### 5b Expertenpruefung der Fehler (Werkzeug steht, Urteil steht aus)
`cmb/review/build_review.py` baut eine Offline-Pruefseite nach dem Vorbild des WMH-Editors `7_qc_edit.py` (eine
HTML-Datei, kein Server, Urteile als JSON, Tastaturbedienung): je Kandidat sieben aufeinanderfolgende axiale Schichten
plus koronar und sagittal, Urteil Blutung / keine / unsicher und bei "keine" der Grund (Gefaess, Verkalkung,
Sulcus-Liquor, Artefakt, Rand). Fuer a03-aniso-e60 ohne Uebermalung: 201 Kandidaten = 53 Fehlalarme, 59 verpasste
Referenz-Blutungen, 89 Treffer (`ergebnisse/review/a03-aniso-e60-gd/review.html`). Auswertung:
`cmb/review/summarise_review.py`. Ziel: wie viel der verbleibenden Luecke ist Annotationsrauschen?
**Verblindete Fassung fuer die Etikett-Decke (21.09. abends, Nutzerfreigabe der Empfehlung 5q a).** *Beobachtung:* die Seite vom 19.09. nennt je Kandidat die Art ("false positive", "missed"), die Netz-Wahrscheinlichkeit und zeichnet
Netz (rot) und Referenz (gelb) ein -- gut zum Erkunden, aber ein Urteil, das die Guete der REFERENZ messen soll, darf all das nicht kennen. *Aenderung:* `build_review.py --blind` (Zufallsreihenfolge mit `--seed`, neutrale Kennungen c001...,
keine Art / Wahrscheinlichkeit / Groesse / Umrisse, stattdessen ein fuer alle gleicher Kreis von 8 mm am Ort des Kandidaten; der Schluessel liegt als `<out>.key.json` NEBEN dem Seitenordner und wird von der Seite nie geladen), `--repeat N`
(N Kandidaten erscheinen zweimal -> Uebereinstimmung des Befunders mit sich selbst) und `--fixed-cell` (Kandidaten aus den Wahrscheinlichkeitskarten am berichteten Arbeitspunkt 0.3 / 2 mm3 / Liquor-Regel statt aus den je Falte gewaehlten
Binaerkarten -- geprueft werden damit genau die Fehler hinter der berichteten Zahl). Vorgabeverhalten unveraendert (Sicherungen `*.vor-blind-0921`). *Gebaut:* `ergebnisse/review/a03-aniso-e60-gd-blind/review.html`, 195 Kandidaten = 54
Fehlalarme + 47 verpasste + 94 Treffer-Komponenten (92 Referenz-Blutungen) + 20 Wiederholungen = 215 Ansichten; Gegenprobe: 2 x 92 / (2 x 92 + 54 + 47) = 0.646 = die berichtete Zahl. Leckpruefung: die Seite enthaelt weder Art noch
Wahrscheinlichkeit noch Umriss-Dateien, die Bildnamen sind neutral; der Neubau mit gleichem Startwert ist bitgleich. *Auswertung, VOR dem ersten Urteil festgelegt* (`summarise_review.py --key`): das Expertenurteil ersetzt die Referenz SYMMETRISCH --
Fehlalarm als Blutung beurteilt -> Treffer; verpasste Referenz als "keine Blutung" beurteilt -> zaehlt nicht mehr; GETROFFENE Referenz als "keine Blutung" beurteilt -> Referenz gestrichen UND der Treffer wird zum Fehlalarm (die Nachbeurteilung
kann dem Netz also auch schaden); "unsicher" und Unbeurteiltes behalten den Status; bei doppelt gezeigten Kandidaten zaehlt die ERSTE Ansicht (Zeitstempel). Ausgabe: F1 / P / S mit der gelieferten und der nachbeurteilten Referenz,
Bootstrap ueber die 57 Faelle, gepaarte Differenz, Uebereinstimmung und Cohens Kappa auf den Wiederholungen. Rechenprobe mit kuenstlichen Urteilen: leer 0.646; alle Fehlalarme = Blutung 0.861 (TP 146, FP 0); alle Verpassten = keine 0.773;
alle Treffer = keine 0.000 (FP 148) -- von Hand nachgerechnet, stimmt. Das sind zugleich die aeussersten Grenzen dessen, was das Urteil bewegen kann. *Grenzen:* ein Befunder, der zugleich Projektbeteiligter ist (die Verblindung mildert das,
hebt es nicht auf); gezeigt wird nur, was Netz ODER Referenz gefunden haben -- was beide uebersehen, bleibt unsichtbar, die nachbeurteilte Sensitivitaet ist eine Obergrenze. **Stand: wartet auf das Urteil des Nutzers.**

## 5e Transfer auf die eigene Kohorte: Versuchsplan (19.09. abends, Laeufe eingereiht)
**Frage:** Welche Strategie liefert auf catalina (T2* 75 Faelle / 830 CMB; SWI 61 / 653) das beste Modell -- und was kostet sie auf VALDO?
**Gemeinsame Grundlage (aus den Befunden dieses Tages):** anisotropes U-Net, 60 feste Epochen, T2*/SWI + FRST, KEINE
Gefaess-Uebermalung, gleiches Gitter 0.5 x 0.5 x 1.0 mm (`cmb/transfer/build_private_grids.py`: 75/75 und 61/61 Faelle gebaut, je
1 Fall mit veraenderter Laesionszahl und 1 mit Laesion ausserhalb des Zuschnitts). Falten: die festen, nach CMB-Zahl
geschichteten Falten der MB-Arena (Klassen 0 / 1-2 / 3-5 / >5; je Falte 5-6 Faelle ohne CMB und 5-6 mit > 5) bzw. die VALDO-Falten.
**Strategien (alle 5-fach, jeder Testfall ungesehen):**
| Kuerzel | Training | misst |
|---|---|---|
| zero-shot | nur VALDO (Ensemble der 5 Faltenmodelle) | Domaenenabfall ohne jede Anpassung; Latte: SHIVA-CMB 0.312 |
| cat | nur catalina | was die eigene Kohorte allein hergibt (Kontrolle) |
| ft | VALDO-Faltenmodell k als Start, Feintuning auf catalina-Falten != k (30 Ep., Lernrate 1e-4) | bringt Vortraining etwas? |
| mix | VALDO + catalina gemeinsam, Test = Falte k beider Kohorten | ein Modell fuer alles; Wirkung auf VALDO |
| mix-alle | VALDO + T2* + SWI | hilft oder stoert die andere Sequenz? |
**Auswertung (`cmb/transfer/evaluate.py`):** fuer ALLE Laeufe dieselbe feste Nachbearbeitung (Schwelle 0.3, >= 2 mm3; auf VALDO
kreuzweise gewaehlt, auf catalina NICHT nachgestellt), je Kohorte gepooltes F1, Precision, Sensitivitaet, FP je Fall, Median-F1 je
Proband, F1 je CMB-Zahl-Klasse, FP in Faellen ohne CMB. Vergleich gepaart je Fall. Summen der privaten Kohorte stehen in mb-arena/.
**Erwartung (vorab festgehalten):** mix >= ft > cat > zero-shot auf catalina; mix kostet auf VALDO nichts oder hilft.
Rauchtest zero-shot an den zwei CMB-reichsten T2*-Faellen: F1 0.71 (P 0.91, S 0.58) -- nur Funktionstest, keine Aussage.
**Erstes Ergebnis, zero-shot auf catalina-T2* (75 Faelle, 828 CMB, feste Nachbearbeitung, ohne Liquor-Regel):** F1 0.622, Precision 0.60,
Sensitivitaet 0.65, Median-F1 je Proband 0.62 -- aber 4.76 Fehlalarme je Fall (3.38 in blutungsfreien Faellen; auf VALDO 1.5).
Nach CMB-Zahl: 1-2: 0.38 | 3-5: 0.54 | > 5: 0.68. SHIVA-CMB ohne Anpassung: 0.312 (P 0.92, S 0.19; andere Nachbearbeitung).
Unser auf 57 VALDO-Faellen trainiertes Netz uebertraegt also deutlich besser als das auf 450 Scans trainierte fremde Modell;
das offene Problem auf catalina sind die Fehlalarme, nicht die Sensitivitaet.
**Reihenfolge (19.09. 22:40 umgestellt):** zuerst die zwei Entscheidungen, die ALLE Transfer-Laeufe betreffen -- Kanaele (mit/ohne FRST)
und Gitter (z = 1 oder 2 mm) --, Entscheidungsregel vorab: die einfachere Variante gewinnt bei gepaartem Delta >= -0.02.
Der Strategievergleich laeuft bewusst mit 60 Epochen; nur die Gewinnerstrategie bekommt am Ende das lange Training.
**Ergebnisse T2* (feste Nachbearbeitung 0.3 / 2 mm3, ohne Liquor-Regel; 75 Faelle, 828 CMB), Stand 20.09. 06:45:**
| Strategie | F1 | Precision | Sensitivitaet | FP je Fall | FP in blutungsfreien Faellen | Median-F1 je Proband | F1 bei 1-2 / 3-5 / > 5 CMB |
|---|---|---|---|---|---|---|---|
| SHIVA-CMB ohne Anpassung (eigene Nachbearbeitung) | 0.312 | 0.92 | 0.19 | 0.17 | -- | 0.42 | -- |
| zero-shot (nur VALDO, 57 Faelle) | 0.622 | 0.60 | 0.65 | 4.76 | 3.38 | 0.62 | 0.38 / 0.54 / 0.68 |
| nur catalina (5-fach) | 0.663 | 0.67 | 0.66 | 3.63 | 1.73 | 0.67 | 0.37 / 0.60 / 0.70 |
| gemischt VALDO + catalina (60 Ep.) | 0.623 | 0.56 | 0.70 | 6.07 | 3.88 | 0.60 | 0.36 / 0.55 / 0.69 |
| Feintuning (VALDO-Start, 30 Ep., Lernrate 1e-4) | 0.657 | 0.63 | 0.69 | 4.43 | 2.92 | 0.67 | 0.47 / 0.57 / 0.71 |
Gemischt, VALDO-Teil derselben Laeufe: F1 0.560 (P 0.52, S 0.60, 1.35 FP je Fall) gegen 0.585 beim reinen VALDO-Training.
**Vorab-Erwartung widerlegt (20.09. 08:50):** gemischtes Training ist bei 60 Epochen auf KEINER der beiden Kohorten besser als das
Training auf der Kohorte allein (catalina 0.623 gegen 0.663; VALDO 0.560 gegen 0.585). Es findet mehr (Sensitivitaet 0.70), meldet aber
deutlich mehr Falsches. Moegliche Ursache, noch nicht geprueft: die Epoche ist auf 800 Patches festgelegt -- bei 132 statt 57 bzw. 75
Faellen sieht das Netz jeden Fall nur halb so oft; ein fairer Vergleich braeuchte gleich viele Durchgaenge JE FALL.
**Gepaart gegen "nur catalina" (75 Faelle, Bootstrap ueber Faelle, 10 000 Ziehungen; `cmb/transfer/evaluate.py`):** zero-shot -0.041
[-0.069; -0.016] (gesichert schlechter) | gemischt -0.040 [-0.077; -0.012] (gesichert schlechter) | Feintuning -0.006 [-0.029; +0.015]
(gleichauf). **Ergebnis T2*:** eigenes Training und Feintuning sind gleich gut; Feintuning braucht die HALBE Trainingszeit (30 statt 60
Epochen) und ist bei Faellen mit nur 1-2 Blutungen besser (0.47 gegen 0.37), macht aber mehr Fehlalarme (4.4 gegen 3.6 je Fall).
Gemischtes Training lohnt bei gleicher Schrittzahl nicht.
**SWI (61 Faelle, 652 CMB), Stand 20.09. 11:50:** zero-shot F1 0.305 (P 0.32, S 0.29, 6.7 FP je Fall) -- das T2*-Netz uebertraegt sich NICHT
auf SWI; nur SWI 5-fach 0.634 (P 0.62, S 0.65, 4.3 FP je Fall, Median 0.67; nach CMB-Zahl 0.51 / 0.60 / 0.70).
**SWI-Feintuning (20.09. 13:25):** F1 0.599 (P 0.62, S 0.58, 3.9 FP je Fall; nach CMB-Zahl 0.50 / 0.57 / 0.65); gepaart gegen nur-SWI
-0.036 [-0.074; +0.003], zero-shot -0.330 [-0.441; -0.220]. Anders als bei T2* (dort gleichauf) ist der T2*-Startpunkt fuer SWI eher
eine Last als eine Hilfe: gleiche Praezision, aber 7 Punkte weniger Sensitivitaet. Deutung (Hypothese): 30 Epochen bei lr 1e-4 reichen
nicht, um ein auf den falschen Kontrast eingestelltes Netz umzulernen; die Epochenfrage (5h) stellt sich hier also umgekehrt.
Datei: mb-arena/transfer/auswertung_swi_alle_0920.json.
Zwischenfazit: Das reine VALDO-Netz kommt ohne einen einzigen catalina-Fall auf 0.62; eigenes Training auf catalina bringt +0.04
und halbiert die Fehlalarme in blutungsfreien Faellen. Faelle mit nur 1-2 Blutungen bleiben in beiden schwach (0.37-0.38).
**Entscheidung 20.09. 02:40:** FRST bleibt (-0.053 ohne), z = 1 mm bleibt (-0.036 mit 2 mm) -> der Transfer-Block laeuft UNVERAENDERT wie eingereiht.

**mix-alle (VALDO + catalina-T2* + catalina-SWI, 193 Faelle, 60 Ep.; 20.09. 17:00):** in ALLEN drei Kohorten gesichert schlechter als das
jeweilige Einzeltraining -- T2* 0.618 (gepaart -0.044 [-0.082; -0.018]), SWI 0.547 (-0.087 [-0.126; -0.048]), VALDO 0.434
(-0.150 [-0.238; -0.054]; zum Vergleich mix-valdo-t2s auf VALDO 0.560, -0.025 [-0.080; +0.040]). Je mehr Kohorten in denselben
60 x 800 Ausschnitten, desto groesser der Verlust, und am meisten verliert die KLEINSTE Kohorte (VALDO: 139 von 1619 Blutungen, die
laesionszentrierte Stichprobe zieht sie entsprechend selten). Das passt zur Budget-Hypothese aus 5h, beweist sie aber nicht: eine
echte Unvertraeglichkeit T2*/SWI wuerde dasselbe Bild geben. Trennen kann das nur der Lauf mit gleichem Budget je Fall (mix e120, eingereiht).
**Zwischenfazit Strategie (Stand 20.09.):** je Sequenz ein eigenes Modell; T2* eigenes Training oder Feintuning (gleichauf), SWI eigenes
Training; Mischen bei festem Budget nicht.

### 5f Reihenfolge Rezept -> langes Training (Nutzerfrage 20.09.)
Das 300-Epochen-Training (6 h) war vor den Rezept-Ablationen eingereiht. Falsch herum: gewinnt danach ein anderes Verhaeltnis
positiv:negativ, Attention, die Perzentil-Skala oder ein groesserer Ausschnitt, muesste der lange Lauf wiederholt werden. Neu: erst alle
60-Epochen-Ablationen (1:2, 1:1, Attention, grosse Ausschnitte, Perzentil-Skala mit zweitem Startwert), dann das Endrezept festlegen,
dann EIN langes Training damit. Der 300-Epochen-Eintrag ist zurueckgestellt (in der Schlange als gestrichen markiert).

### 5g Verhaeltnis positiv:negativ als Kette (Nutzerwunsch 20.09., Regel vorab festgelegt)
Nach den Laeufen 1:2 und 1:1 prueft `cmb/analysis/ratio_chain.py` die Ergebnisse und reiht selbststaendig den naechsten Schritt ein:
solange das gepoolte F1 des neuesten Verhaeltnisses ueber dem bisher besten liegt, geht es weiter zu 1:3, 1:4, 1:5, hoechstens 1:6;
beim ersten Schritt ohne Gewinn ist Schluss. Hilft weder 1:1 noch 1:2, wird einmal die Gegenrichtung 2:1 geprueft. Jeder Schritt
wird mit gepaartem Intervall ins Protokoll geschrieben; die Kurve als Ganzes ist das Ergebnis, nicht der einzelne Schritt.
**Ergebnis 1:2 (21.09. 06:40): das Training BRICHT ZUSAMMEN.** Mit 33 % Laesions-Ausschnitten kippt das Netz in die leere Vorhersage: Falte 0 inneres F1 0.000 von Epoche ~10 bis 60, Verlust
steht bei 0.178, Test-F1 0.091 (bestes Modell = Epoche 5); Falte 1 dasselbe ab Epoche 15 (0.187 -> 0.178). Der Plateauwert ist genau der Verlust von "nichts vorhersagen": Dice-Anteil 0.5 x Anteil
der Ausschnitte mit Blutung (0.33) = 0.167 plus ein kleiner BCE-Rest -- fuer leere Ausschnitte belohnt der geglaettete Dice die leere Vorhersage mit Verlust 0. Bei 60 % entkommt das Training
diesem Minimum, bei 33 % nicht. Lauf nach zwei von fuenf Falten von mir abgebrochen (zwei von zwei kollabiert; die GPU geht an den Gewebekanal) -- Ordner `a03-aniso-e60-l33-gd` enthaelt Falte 0
vollstaendig, Falte 1 unvollstaendig. **Folge:** weniger Positive ist keine Option; die Kette Richtung 1:3, 1:4 ... entfaellt damit von selbst (Regel: nur weiter, wenn besser). Der Lauf 1:1 bleibt
eingereiht -- er zeigt, wo die Kante zwischen 60 % (stabil) und 33 % (Kollaps) liegt. Die Gegenrichtung 2:1 reiht `ratio_chain.py` wegen des unvollstaendigen 1:2-Laufs NICHT mehr selbst ein;
falls gewuenscht, von Hand. **Nebenbefund fuer die Methode:** das Rezept ist in diesem Punkt empfindlich -- der Anteil der Laesions-Ausschnitte ist kein harmloser Regler, sondern haelt das Training stabil.

### 5h Haengt die Strategiewahl an der Epochenzahl? (Nutzerfrage 20.09. mittags)
**Beobachtung** aus den gespeicherten Lernkurven (`verlauf.json`, inneres F1, Mittel der 5 Falten; T2*):
nur-catalina e60: Ep. 30 0.624 -> 50 0.653 -> 60 0.657 (flach ab 50). gemischt e60: 30 0.602 -> 50 0.636 -> 55 0.648 -> 60 0.645
(liegt von Ep. 25 bis 45 um 0.02-0.03 zurueck, holt am Ende auf 0.012 auf). Feintuning e30: schon nach 2 Epochen 0.629, ab Ep. 16 ~0.66.
**Grenzen dieser Beobachtung:** (1) die innere Falte des gemischten Laufs enthaelt auch VALDO-Faelle, die Kurven sind also nicht
fallgleich; (2) der Kosinus-Plan faehrt die Lernrate auf null, JEDE Kurve wird am Ende flach -- "flach" beweist keine Konvergenz;
(3) inneres F1 mit Trainingsschwelle, nicht die feste Nachbearbeitung der Tabelle.
**Hypothese:** der Rueckstand des gemischten Trainings ist (teilweise) ein Budget-Effekt: eine Epoche sind fest 800 Ausschnitte, bei
57+75 statt 75 Faellen sieht das Netz jeden catalina-Fall nur ~0.57-mal so oft. Feintuning dagegen ist frueh fertig und duerfte mit
weniger Epochen auskommen.
**Experiment (eingereiht 20.09. 12:20):** gemischt e120 (gleich viele Durchgaenge je catalina-Fall wie nur-catalina e60), nur-catalina
e120 als Kontrolle (sonst waere "mehr Training" mit "Mischen" vermengt), Feintuning e10 (vorgezogen, ~25 min). Auswertung wie 5e:
feste Nachbearbeitung, gepaart gegen nur-catalina e60. **Entscheidungsregel vorab:** Mischen gilt nur dann als rehabilitiert, wenn
gemischt e120 gepaart nicht schlechter ist als nur-catalina e120 (nicht nur als e60).

**Nutzerfrage 21.09. ("alles mischen -- der Datensatz ist dann groesser"): Stand und zwei neue Laeufe.** Gemessen ist bisher nur das Mischen bei GLEICHEM Budget (60 Ep. x 800 Ausschnitte):
VALDO + catalina-T2* (132 Faelle) -0.040 auf catalina, -0.025 n. s. auf VALDO; alle drei Kohorten (193 Faelle) -0.044 (T2*), -0.087 (SWI), -0.150 (VALDO) -- je mehr Kohorten, desto schlechter, am
staerksten fuer die kleinste. Das kann am Budget liegen (jeder Fall wird seltener gesehen; die laesionszentrierte Stichprobe zieht VALDO mit 139 von 1619 Blutungen nur zu 9 %) oder an echter
Unvertraeglichkeit (andere Masken, Schichtdicken, SWI-Kontrast, Annotationsstil). Zwei Wege, die groessere Datenmenge trotzdem zu nutzen: (a) gleiches Budget je Fall -- gemischt mit 120 Ep. gegen
eigenes Training mit 120 Ep. (eingereiht seit 20.09., jetzt vorgezogen); (b) **erst auf ALLEM vortrainieren, dann auf die Zielkohorte feintunen** -- der uebliche Weg, grosse gemischte Daten zu
nutzen: Start von den mix-alle-Faltenmodellen (193 Faelle), 30 Ep., lr 1e-4, einmal auf catalina-T2*, einmal auf catalina-SWI (dort mit `manifest_swi_mix.json`, damit die Falten zu mix-alle
passen und kein Testfall im Vortraining lag). Vergleich gepaart gegen eigenes Training (T2* 0.663, SWI 0.634) und gegen Feintuning vom reinen VALDO-Netz (0.657 / 0.599). **Erwartung vorab:**
(b) ist auf SWI besser als das Feintuning vom T2*-Netz (das Vortraining kennt jetzt SWI); ob es das eigene Training schlaegt, ist offen.
**Ergebnis (b) auf catalina-T2* (21.09. 10:00):** Feintuning vom Alles-Modell F1 0.623 (P 0.58, S 0.67, 5.3 Fehlalarme je Fall) -- gesichert SCHLECHTER als eigenes Training (-0.039 [-0.083; -0.007]) und als
Feintuning vom reinen VALDO-Netz (-0.033 [-0.066; -0.009]). Das gemischte Vortraining ist fuer T2* also der schlechtere Startpunkt; 30 Epochen bei lr 1e-4 holen den Rueckstand des Alles-Modells (0.618)
nicht auf. Der SWI-Lauf (dort war die Erwartung positiv) steht noch aus.
**Ergebnis (b) auf catalina-SWI (21.09. 13:50):** Feintuning vom Alles-Modell F1 0.590 (P 0.58, S 0.60, 4.7 Fehlalarme je Fall) -- gesichert schlechter als eigenes SWI-Training (-0.045 [-0.073; -0.021]) und nicht besser
als das Feintuning vom reinen T2*-Netz (0.599). **Meine Erwartung vorab war falsch:** dass das Vortraining SWI schon kennt, hilft nicht. Damit ist das Bild fuer beide Sequenzen gleich -- gemischtes Vortraining ist der
schlechtere Startpunkt; das passt zu 5n (was KEINE Blutung ist, uebertraegt sich nicht zwischen den Kohorten). Offen bleibt nur noch die Budget-Frage (gemischt 120 Ep. gegen eigenes Training 120 Ep., eingereiht).
**Ergebnis (a), erster Teil -- gemischt mit 120 Epochen (21.09. 19:30; gewaehlte Zwischenstaende 100 / 100 / 90 / 110 / 100):** catalina-T2* F1 0.641 (P 0.58, S 0.71, 5.6 Fehlalarme je Fall) gegen gemischt 60 Ep. 0.623:
gepaart **+0.019 [+0.003; +0.037], gesichert** -- die Budget-Vermutung des Nutzers stimmt in der Richtung. Gegen eigenes Training mit 60 Ep. (0.663): -0.021 [-0.054; +0.006], nicht mehr gesichert schlechter (mit 60 Ep. war es
-0.040 gesichert). VALDO-Anteil: 0.582 gegen nur-VALDO 0.585, gepaart -0.003 [-0.066; +0.056] (mit 60 Ep. -0.025). Das Mischen holt mit doppeltem Budget also auf: auf VALDO gleichauf, auf catalina noch 0.02 zurueck, vor allem
ueber die Fehlalarme (5.6 gegen 3.6 je Fall). **Noch nicht entschieden:** nach der vorab festgelegten Regel zaehlt der Vergleich gegen das eigene Training mit EBENFALLS 120 Epochen (eingereiht) -- profitiert auch das von der Laenge,
bleibt der Abstand.
**Ergebnis (a), zweiter Teil und ENTSCHEIDUNG -- eigenes Training mit 120 Epochen (22.09. 04:09; gewaehlte Zwischenstaende 70 / 60 / 40 / 90 / 120; `mb-arena/transfer/auswertung_t2star_cat_e120_0922.json`,
`auswertung_t2star_mix_vs_cat_e120_0922.json`, `operating_point_t2star_cat_e120_0922.json`):**
| catalina-T2*, feste Nachbearbeitung | F1 | P | S | FP je Fall | gepaart gegen eigenes Training 60 Ep. |
|---|---|---|---|---|---|
| eigenes Training 60 Ep. | **0.663** | 0.67 | 0.66 | 3.6 | -- |
| eigenes Training 120 Ep. | 0.640 | 0.60 | 0.69 | 5.0 | **-0.022 [-0.052; -0.002], gesichert schlechter** |
| gemischt 120 Ep. | 0.641 | 0.58 | 0.71 | 5.6 | -0.021 [-0.054; +0.006] |
| gemischt 60 Ep. | 0.623 | 0.56 | 0.70 | 6.1 | -0.040 [-0.077; -0.012] |
Vorab festgelegter Vergleich: gemischt 120 gegen eigenes 120 = **+0.001 [-0.021; +0.023] -- gleichauf.** Mit Liquor-Regel: eigenes 120 Ep. 0.656 gegen 0.675 (kreuzweise -0.016 [-0.033; +0.000]).
**Entscheidung nach der Regel:** Mischen ist bei GLEICHEM BUDGET JE FALL rehabilitiert -- es kostet dann nichts mehr. Die beste Strategie aendert das NICHT: das eigene Training mit 60 Epochen bleibt vorn (0.663 / 0.675), das gemischte Netz
liegt -0.021 (n. s.) darunter und ist auf VALDO gleichauf (0.582 gegen 0.585). Ein Modell fuer beide Kohorten ist also moeglich, zum Preis von ~0.02 auf catalina, nicht gesichert.
**Was die vier Laeufe zusammen sagen -- es geht um das Budget je Fall, nicht um Epochen:** eine Epoche sind fest 800 Ausschnitte, also sieht ein Trainingsfall bei eigenem Training 60 Ep. ~1070 Ausschnitte (45 Trainingsfaelle je Falte),
gemischt 60 Ep. ~610 (79 Faelle), gemischt 120 Ep. ~1210, eigenes 120 Ep. ~2130. Rangfolge nach F1: 1070 (0.663) > 1210 (0.641) ~ 2130 (0.640) > 610 (0.623). Zu wenig Budget kostet (gemischt 60), zu viel kostet ebenfalls (eigenes 120: Precision
0.67 -> 0.60, Fehlalarme 3.6 -> 5.0 -- dasselbe Muster wie die spaeten Zwischenstaende in 5o), und die Zwischenstandswahl faengt das nicht auf, weil der Kosinus-Plan mit 120 Ep. gestreckt ist (Epoche 40 von 120 ist ein anderer Zustand als
Epoche 40 von 60). **Meine Erwartung vom 20.09. war halb richtig:** der Rueckstand des Mischens war ein Budget-Effekt, aber "mehr Budget" ist keine Einbahnstrasse.
**Vorhersage fuer VALDO mit 120 Ep. (laeuft, festgehalten VOR dem Ergebnis):** VALDO hat mit 60 Ep. schon ~1400 Ausschnitte je Trainingsfall (33-36 Faelle je Falte), also mehr als catalina; 120 Ep. = ~2800. Nach der Budget-Deutung
darf 120 Ep. dort NICHT helfen und wird eher schaden (mehr Fehlalarme). Trifft es zu, ist "120 Epochen generell?" beantwortet: nein -- die richtige Groesse ist Ausschnitte je Fall, ~1000-1400 in diesem Rezept; mehr Faelle brauchen mehr Epochen,
mehr Epochen bei gleichen Faellen nicht. Hilft es doch, ist die Budget-Deutung falsch und VALDO braucht eine andere Erklaerung (kleinere Faelle, mehr Hintergrund je Ausschnitt).
**Ergebnis VALDO 120 Ep. (22.09. 06:50; gewaehlte Zwischenstaende 100 / 100 / 90 / 110 / 90; `ergebnisse/analysis/valdo_e120_gepoolt.json`, `valdo_e120_split.json`, `operating_point_valdo_e120.json`): die Vorhersage trifft zu.**
Feste Zelle roh 0.556 gegen 0.585 (gepaart -0.029 [-0.068; +0.019]), P 0.47 gegen 0.52, S 0.68 gegen 0.67, Fehlalarme je Fall 1.84 gegen 1.51; mit Liquor-Regel 0.610 gegen 0.646, kreuzweise 0.592 gegen 0.622 (-0.031 [-0.078; +0.018]).
Je Kohorte: 4 mm +0.022 (n = 8, n. s.), 0.8 mm **-0.046 [-0.074; -0.014]**, 3-4 mm -0.015 (n. s.). Auf VALDO allein ist "schlechter" nicht gesichert (57 Faelle; der Abstand liegt in der Groesse des Startwert-Rauschens), "nicht besser" schon --
und der Mechanismus ist derselbe wie auf catalina: mehr Empfindlichkeit, weniger Precision, mehr Fehlalarme. **Antwort auf die Nutzerfrage "120 Epochen generell besser?": nein.** Drei gepaarte Messungen (gemischt +0.019 gesichert, eigenes
catalina -0.022 gesichert, VALDO -0.029 n. s.) sind mit EINER Groesse erklaert -- Ausschnitte je Trainingsfall; 120 Epochen helfen nur dort, wo 60 zu wenige waren (gemischt, ~610 je Fall), und schaden, wo 60 schon ~1000-1400 je Fall waren.
Fuer "schwierigere Netze" (nnU-Net-Bauart, Attention) ist das nicht gemessen; die Herausforderer-Laeufe (5s) laufen deshalb absichtlich mit 60 Ep. **Grenzen:** ein Startwert je Lauf; die Regel "~1000-1400 Ausschnitte je Fall" ist aus vier
Punkten abgelesen, kein feines Optimum; nicht getestet ist die naheliegende Gegenprobe "60 Ep. mit 400 statt 800 Ausschnitten" (halbes Budget), die zeigen wuerde, ob auch VALDO schon UEBER dem Optimum liegt.
**Nutzerfrage 21.09. abends: "sind 120 Epochen generell besser, besonders bei schwierigeren Netzen und mehr Daten?"** Direkt gemessen ist bisher nur der Mischlauf (+0.019 gesichert). Zwei Hinweise aus vorhandenen Laeufen: (1) der
gewaehlte Zwischenstand liegt umso haeufiger am Planende (>= Epoche 55), je mehr Daten im Training sind -- VALDO 2 bzw. 1 von 5 Falten, catalina-T2* und -SWI je 3 von 5, gemischt 4 von 5; wer am Planende noch gewinnt, war noch
nicht fertig. (2) Eine "Epoche" sind bei uns fest 800 Ausschnitte; 60 Epochen sind 12 000 Schritte -- die VALDO-Spitze (nnU-Net: 1000 x 250 Schritte) und CenSynCMB (200 Epochen) trainieren um ein Vielfaches laenger. Massgeblich
ist also die Zahl der Schritte im Verhaeltnis zu Datenmenge und Netzgroesse, nicht die Epochenzahl an sich. **Offene Flanke:** das Architektur-Turnier (4.4) lief mit 40 Epochen und Fruehstopp -- grosse Netze (Residual-SE, U-Net++,
SegResNet, Swin) koennten dadurch benachteiligt gewesen sein; die Rangfolge gilt nur "bei diesem Budget". **Eingereiht:** Basisrezept auf VALDO mit 120 Epochen (neben dem Kontrolllauf nur-catalina 120 Ep.).
Erwartung vorab: auf VALDO allein (57 Faelle, Zwischenstaende meist vor dem Planende) wenig bis kein Gewinn, auf catalina ein kleiner.

**Erstes Ergebnis zu 5h (20.09. 14:30): Feintuning 10 gegen 30 Epochen, T2*.** ft10 F1 0.619 (P 0.56, S 0.69, 6.0 FP je Fall), ft30 0.657
(P 0.63, S 0.69, 4.4); gepaart gegen nur-catalina: ft10 -0.044 [-0.090; -0.003], ft30 -0.006 [-0.030; +0.015]. Die Sensitivitaet ist nach
10 Epochen schon da, die zusaetzlichen Epochen bauen FEHLALARME ab. Meine Lesart der Lernkurve ("nach 2 Epochen fast fertig") war
damit falsch -- das innere F1 mit Trainingsschwelle zeigte den Unterschied nicht. Folge: Feintuning mit 60 Epochen eingereiht (T2* und
SWI); offen ist jetzt, ob Feintuning bei gleicher Epochenzahl das eigene Training UEBERHOLT.

### 5i Wo liegen Fehlalarme und Uebersehene? (Idee aus der Methodenrecherche, 20.09.; `cmb/analysis/fp_location.py`)
**Anlass:** Arbeiten zur CMB-Detektion senken Fehlalarme mit Ortswissen (arXiv:2306.13020, dort 14.7 -> 0.56 FP je Proband, ungeprueft).
Wir nutzen bisher nur die Liquor-Regel. **Messung:** Ort je Vorhersage und je Referenzblutung aus der SynthSeg-Karte (Mehrheitslabel),
feste Nachbearbeitung wie in 5e. **VALDO (a03-aniso e60, 57 Faelle):**

| Ort | Vorhersagen | TP | FP | Precision | Referenz | gefunden | Sensitivitaet |
|---|---|---|---|---|---|---|---|
| lobaer | 84 | 48 | 36 | 0.57 | 78 | 48 | 0.62 |
| tief | 44 | 33 | 11 | 0.75 | 38 | 31 | 0.82 |
| infratentoriell | 19 | 13 | 6 | 0.68 | 15 | 13 | 0.87 |
| mesiotemporal | 0 | 0 | 0 | -- | 3 | 0 | 0.00 |
| Liquor | 33 | 1 | 32 | 0.03 | 5 | 1 | 0.20 |

Was-waere-wenn (explorativ, NICHT kreuzweise gewaehlt): Liquor streichen 0.585 -> 0.646 (bekannt, 4.3); jeder andere Ort gestrichen oder
mit strengerer Schwelle (0.5/0.7/0.9) VERSCHLECHTERT das F1. **catalina-T2* (Summen; nur-catalina / Feintuning):** lobaer Precision
0.71 / 0.66, Sensitivitaet 0.71 / 0.74; tief 0.67 / 0.68 und 0.79 / 0.79; **infratentoriell Precision 0.76 / 0.74, aber Sensitivitaet nur
0.46 / 0.50** bei 207 von 828 Blutungen -- 111 von 283 Uebersehenen (39 %) liegen in Kleinhirn und Hirnstamm. Liquor-Regel bringt dort
nur +0.008 bis +0.014 (VALDO +0.06), weil SynthSeg auf 5-mm-T2* 24 echte Blutungen in den Liquor legt.
**Ergebnis:** Ortswissen ueber die Liquor-Regel hinaus senkt unsere Fehlalarme NICHT -- die Fehlalarme sitzen dort, wo auch die
Blutungen sitzen (lobaer). Der Befund der Analyse ist ein anderer: auf catalina ist die infratentorielle SENSITIVITAET die groesste
einzelne Luecke. **Hypothesen:** 5-mm-Schichten im Kleinhirn (Partialvolumen), dunkle Nachbarstrukturen (Nucleus dentatus, Sinus),
enge HD-BET-Maske am Unterrand; VALDO hat nur 15 infratentorielle Blutungen, das vortrainierte Netz kennt die Region kaum.
**Naechster Schritt:** Uebersehene infratentorielle Blutungen nach Groesse/Kontrast/Randabstand aufschluesseln (CPU), dann ggf.
ortsbewusste Stichprobe (infratentorielle Blutungen haeufiger ziehen).

**KORREKTUR (20.09. 17:10, `cmb/analysis/missed_lesions.py`): die "infratentorielle Luecke" ist ein FALL-Effekt, kein Orts-Effekt.**
Die Aussage oben ("groesste einzelne Luecke", Hypothesen 5-mm-Schichten / Nucleus dentatus / enge Maske) war voreilig: ich hatte gepoolte
Laesionszahlen gelesen, ohne zu pruefen, wie sie sich auf Faelle verteilen. Nachgemessen (nur-catalina, 75 Faelle, 828 Blutungen):
- 25 Faelle tragen infratentorielle Blutungen; die DREI Faelle mit den meisten Uebersehenen halten 128 von 207 dieser Blutungen und
  95 von 111 Uebersehenen (zusammen 234 aller 828 Blutungen). In diesen drei: infratentoriell Sensitivitaet 0.26 (80 von 128 kleiner
  als 10 mm3), lobaer 0.53. In den UEBRIGEN 72 Faellen: infratentoriell **0.80**, lobaer 0.76, tief 0.79 -- kein schwacher Ort.
- Masken-Hypothese widerlegt: 0 infratentorielle Blutungen liegen ausserhalb der Hirnmaske, Randabstand uebersehen/gefunden 11.0 / 12.7 mm.
- Was die Uebersehenen unterscheidet, ist GROESSE und KONTRAST: uebersehen Median 5.0 mm3 (rund 2 native Voxel bei 0.69 x 0.69 x 5 mm)
  und Kontrast 0.137, gefunden 19.4 mm3 und 0.241. Schichtung nach Groesse x Kontrast (lobaer | infratentoriell): klein/schwach
  0.27 | 0.07, klein/stark 0.73 | 0.31, mittel 0.66-0.86 | 0.56-0.77, gross 0.79-0.94 | 0.85-0.91. Bei lobaeren Raten je Schicht waere
  infratentoriell 0.61 zu erwarten, beobachtet 0.46 -- der Rest steckt in den kleinen Blutungen der drei Faelle.
- Das Netz ist an den Uebersehenen nicht "knapp dran": hoechste Wahrscheinlichkeit < 0.05 bei 92 von 111 (infratentoriell) und 107 von
  150 (lobaer). Eine niedrigere Schwelle nur im Kleinhirn/Hirnstamm (0.2 / 0.1 / 0.05, explorativ) SENKT das F1 (0.663 -> 0.660 / 0.650 /
  0.649; Feintuning 0.657 -> 0.650 / 0.640 / 0.638). Struktur: Kleinhirnrinde 0.35, Kleinhirn-Mark 0.57, Hirnstamm 0.43 (von den drei Faellen gepraegt).
**Entscheidung:** ortsbewusste Stichprobe gestrichen. Der Hebel ist die Klasse "klein und kontrastarm" (176 von 828 Blutungen, Sensitivitaet
0.27 / 0.07) -- genau dort setzt der laesionsweise Verlust an (5j). **Offene Frage an den Experten:** sind die vielen 1-2-Voxel-Markierungen
im Kleinhirn dieser drei Faelle sichere Mikroblutungen? (Fallkennungen nur privat: mb-arena/transfer/missed_lesions_t2star_0920_PRIVAT_faelle.json.)
**Gegenprobe ueber die QC-Seite (17:15):** die private Kontrollseite (mb-arena/qc/t2star-nur-catalina, 75 Faelle, Summe TP 545 / FP 272 /
FN 283 = exakt die Auswertung) zeigt fuer die drei Faelle 112 / 78 / 44 Referenzblutungen mit 73 / 61 / 11 Uebersehenen -- ZWEI Patienten tragen
134 der 283 Uebersehenen (47 %). Ohne die drei Faelle: Sensitivitaet 0.77 statt 0.66, Precision 0.64, F1 0.696 statt 0.663 (72 Faelle, 594 Blutungen).
Beide Zahlen gehoeren in einen Bericht; welche die Hauptzahl ist, haengt am Expertenurteil ueber die Kleinstmarkierungen.
**Lehre fuer die Methode:** gepoolte Laesionsmasse auf catalina werden von wenigen Faellen mit sehr vielen Blutungen getragen (3 Faelle = 28 %
aller Blutungen); jede Aufschluesselung nach Ort, Groesse o. ae. braucht die Gegenprobe "ohne die groessten Faelle".

### 5j Laesionsweiser Verlust (blob loss), eingereiht 20.09.
**Original/Ausgangslage:** BCE + Dice ueber den ganzen Ausschnitt. **Beobachtung:** die Klasse mit 1-2 Blutungen ist auf allen Kohorten
die schwaechste (0.37-0.51), kleine Blutungen gehen im globalen Dice unter. **Aenderung:** `--verlust blob` in `code/turnier.py`
(Kofler et al., MICCAI 2023, arXiv:2205.08209): global + 0.5 x Mittel der Verluste je Blutung, jeweils OHNE die Voxel der anderen
Blutungen (entspricht alpha=2, beta=1); 26er-Komponenten, float32, Blob-Term nur am Hauptkopf. **Pruefung (CPU):** leerer Ausschnitt ==
globaler Verlust; perfekte Vorhersage -> 0; eine uebersehene 8-Voxel-Blutung neben einer 72-Voxel-Blutung kostet 0.091 statt 0.014.
**Experiment:** a03-aniso, 60 Ep., gitter_d, sonst unveraendert; Vergleich gepaart gegen 0.588-Basis, zusaetzlich Sensitivitaet nach
Laesionsgroesse. Der protokollierte Verlustwert ist mit anderen Laeufen NICHT vergleichbar (andere Skala), das innere F1 schon.
**Zwischenstand 20.09. 17:55 (4 von 5 Falten, feste Nachbearbeitung, gepaart auf denselben 46 Faellen):** blob - Basis = -0.057 [-0.132; +0.003].
Der Verlust tut, was er soll -- und bezahlt dafuer: Sensitivitaet 0.67 -> 0.70 (TP 74 -> 81 von 115), aber Fehlalarme 58 -> 103. Je Falte
(Basis | blob, TP/FP/FN): F0 23/22/9 | 26/28/6; F1 15/17/11 | 14/16/12; F2 26/13/13 | 26/12/13; F3 10/6/8 | 15/47/3. Falte 3 traegt den
Verlust: 47 Fehlalarme, ueber viele Faelle verteilt (9, 7, 6, 4, ...), 35 davon mit p >= 0.5 -- also kein Schwellenproblem, das Modell dieser Falte
feuert insgesamt mehr. Nutzerhinweis (berechtigt): in 2 von 4 Falten ist blob leicht vorn; +0.02 in einer Falte ist aber ~1 Blutung bei 18-39
Blutungen je Falte, das Rauschmass je Falte liegt bei 0.06. **Naechste Schritte (Nutzerwunsch: Wiederholungen, VALDO UND catalina):**
blob auf catalina-T2* (828 Blutungen, gepaartes Intervall dort rund +-0.025 statt +-0.06 -- ein Lauf sagt mehr als mehrere VALDO-Wiederholungen)
und blob mit Startwert 1 auf VALDO (Basis mit Startwert 1 liegt vor: 0.585); weitere Startwerte gestaffelt, falls das Bild unklar bleibt.
Fuer einen fairen Vergleich zusaetzlich: Schwelle/Mindestgroesse fuer das blob-Modell kreuzweise neu waehlen (die feste 0.3 / 2 mm3 wurde am
Basis-Modell gewaehlt).
**Endstand VALDO 5/5 Falten (20.09. 18:05; `cmb.transfer.evaluate`, `cmb.analysis.missed_lesions`, `cmb.analysis.fp_location`):**
F1 0.540 gegen 0.585, gepaart -0.044 [-0.105; +0.003]; Sensitivitaet 0.67 -> 0.72, Precision 0.52 -> 0.43, Fehlalarme je Fall 1.51 -> 2.30 (86 -> 131).
Sensitivitaet nach Groesse (Basis | blob): klein < 7.2 mm3 (n=44) 0.52 | **0.64**, mittel (n=48) 0.69 | 0.73, gross (n=47) 0.79 | 0.79; Kontrast unter
Median 0.51 | 0.55, darueber 0.83 | 0.89. Der Verlust erreicht also genau die Blutungen, fuer die er gebaut ist. 43 der 131 Fehlalarme liegen im Liquor.
**Arbeitspunkt je Modell, kreuzweise gewaehlt, mit Liquor-Regel (`cmb/analysis/operating_point.py`, neu):** Basis feste Zelle 0.646, kreuzweise 0.622
(Zellen je Falte alle bei Schwelle 0.3); blob feste Zelle 0.599, beste Einzelzelle 0.6 / 2 mm3 = 0.626 (optimistisch), kreuzweise **0.576**
(Zellen streuen: 0.5/4, 0.6/1, 0.6/2, 0.3/4, 0.4/4). Gepaart kreuzweise -0.046 [-0.104; +0.018]. Das blob-Modell ist anders kalibriert (Optimum bei
0.5-0.6 statt 0.3), aber die Neuwahl rettet es nicht -- die Wahl ist zwischen den Falten instabil.
**Zweite Stufe auf den blob-Kandidaten (Nutzer 20.09.: "bei der ersten Stufe sind mehr Fehlalarme nicht schlimm, die zweite soll sie wegnehmen"):**
Als erste Stufe ist blob BESSER: Kandidaten (Schwelle 0.15, Liquor-Regel) erreichen 107 von 139 Blutungen (Obergrenze 0.77) gegen 100 (0.72) beim
Basis-Modell; bei Schwelle 0.10 / 0.05: 109 / 110 (0.78 / 0.79) mit 339 / 371 Kandidaten. 29 Blutungen (21 %) erreicht auch bei 0.05 kein Kandidat.
Geschachtelte Auswertung (CPU-Klassifikatoren): nur Schwelle 0.608 | logistische Regression **0.623** (P 0.63, S 0.62, 0.89 FP je Fall) | Wald 0.614;
bei Kandidatenschwelle 0.05: 0.608 | 0.624 | 0.604. Erstmals schlaegt die zweite Stufe die reine Schwelle (+0.015; "contrast" ist jetzt drittwichtigstes
Merkmal), die Kette blob + Stufe 2 liegt damit GLEICHAUF mit der Basis-Kette (0.622-0.627), nicht darueber: die zweite Stufe nimmt die Fehlalarme weg
(2.96 -> 0.89 je Fall), verliert dabei aber die gewonnene Sensitivitaet (0.77 -> 0.62). Engpass ist jetzt Stufe 2 (115 positive Trainingskandidaten).
**Gegenprobe catalina-T2*, Basis-Modell nur-catalina (20.09. 18:40, CPU-Klassifikatoren, geschachtelt):** 961 Kandidaten (582 auf einer Blutung, 379 falsch)
erreichen 571 von 828 Blutungen -- Obergrenze nur 0.69. Alle annehmen 0.642 | nur Schwelle **0.670** | logistische Regression 0.658 | Wald 0.657. Mit FUENFMAL so vielen
Beispielen schlaegt die zweite Stufe die reine Schwelle also weiterhin NICHT. Die Deutung vom 19.09. ("zu wenige Kandidaten") reicht damit nicht aus: handgemachte Merkmale
sehen nichts, was die erste Stufe nicht schon verrechnet hat. Eine zweite Stufe lohnt nur, wenn sie ANDERE Information bekommt (groesserer Kontext, andere Ansichten,
Uebereinstimmung mehrerer Modelle) -- und der groessere Hebel auf catalina ist die Obergrenze der ersten Stufe (31 % der Blutungen erreicht kein Kandidat).
**Recherche zur zweiten Stufe (20.09. abends, `recherche/zweite-stufe-fp-reduktion.md`; ein Agent, leichtes Modell, 23 Abrufe -- ueber dem Budget von 15;
fast alle Zahlen dort aus Suchausschnitten, als "unverified" markiert) und was davon sofort geprueft wurde:**
- *Gradient Boosting auf denselben 10 Merkmalen* (billigster Vorschlag): VALDO-blob 0.609 (logistisch 0.623, nur Schwelle 0.608), catalina-Basis 0.655 (nur Schwelle 0.670).
  Kein Gewinn -- bestaetigt: nicht die Art des Klassifikators begrenzt, sondern der Informationsgehalt der Merkmale.
- *Ensemble ueber Startwerte* (`cmb/analysis/ensemble.py`, neu; Mittel der out-of-fold-Karten, Falten identisch): Basis-Rezept, Startwerte 42 + 1 auf VALDO. Feste Zelle:
  0.594 gegen 0.585 / 0.553 der Mitglieder, Fehlalarme je Fall 1.30 gegen 1.51 / 1.42. Mit Liquor-Regel und kreuzweise gewaehltem Arbeitspunkt: **0.647** (P 0.65, S 0.64,
  0.82 FP je Fall) gegen 0.622 / 0.564; gepaart gegen Startwert 42 +0.025 [-0.024; +0.079], gegen das Mittel der Mitglieder +0.054. Nebenwirkung: der Arbeitspunkt wird
  STABIL (alle fuenf Falten waehlen 0.3 / 2 mm3; die Einzelmodelle streuen). Erste Fehlalarm-Strategie mit durchgehend positivem Bild; mit nur zwei Mitgliedern nicht gesichert.
  Folge: Startwert 2 fuer Basis und blob eingereiht (zugleich die vom Nutzer gewuenschten Wiederholungen).
- *Ensemble, Fortsetzung (20.09. 19:15): MEHR Mitglieder setzen den Gewinn NICHT fort.* Aus vorhandenen Laeufen gleicher Falten gebaut (Startwerte 42 und 1, Perzentil-Skala,
  drei Uebermalungsstufen, blob). Feste Zelle 0.3 / 2 mm3 mit Liquor-Regel -- Einzelmodelle: s42 0.646, s1 0.590, np995 0.607, ge 0.610, gf 0.612, gg 0.591, blob 0.599
  (Mittel der sechs Basis-nahen: 0.609). Ensembles: 2 Mitglieder 0.647 | 3 0.648 | 5 0.630 | 6 0.639 | s42+blob 0.625 | s42+s1+blob 0.631 | alle sieben 0.644. Also: Ensemble ~
  bestes Mitglied, +0.02 bis +0.04 ueber dem MITTLEREN Mitglied, flach ab zwei Mitgliedern; blob als Mitglied bringt nichts. Der Startwert 42 ist ein Gluecksfall (0.646 gegen
  0.59-0.61 der anderen) -- die ehrliche Erwartung fuer EIN Modell dieses Rezepts liegt bei ~0.61, das Ensemble hebt sie verlaesslich auf ~0.64. Meine Formulierung von 18:55
  ("durchgehend positives Bild") war zu stark. Die kreuzweise gewaehlten Arbeitspunkte (0.59-0.65) schwanken staerker als die Modelle selbst: bei 57 Faellen kostet die Wahl
  unter 40 Zellen je Falte 0.02-0.05 F1 -> fuer Entscheidungen die feste, vorab gewaehlte Zelle benutzen. catalina-T2*: nur-catalina + Feintuning gemittelt 0.669 gegen 0.663,
  gepaart +0.007 [-0.007; +0.021]; mit gemischt als drittem Mitglied 0.662. JSON: ergebnisse/analysis/operating_point_valdo_{ensembles,members}.json.
- *Nicht passend fuer uns:* Focal Loss / OHEM (unsere Kandidatensaetze sind NICHT unbalanciert: 115:169 bzw. 582:379), Kaskade aus zwei Klassifikatoren (der zweite haette noch
  weniger Beispiele), Phase/QSM (keine Daten).
- *Offen und aussichtsreich, weil sie der zweiten Stufe ANDERE Information geben:* Spiegel-TTA (als Ensemble und als Konsistenz-Merkmal), Kontext in mehreren Groessen + Lage
  (Ghafoorian 2017, Lakunen), vortrainierte Encoder bzw. 2.5D-Mehrfachansichten (Setio 2016), synthetische Blutungen UND synthetische harte Negative.
- **Neuer Stand der Technik, an der Quelle geprueft (arXiv 2607.05325, 06.07.2026): CenSynCMB** (He, ..., Barkhof, Davies, Sudre -- die VALDO-Organisatoren): 3D-Attention-U-Net +
  Zentrumskarten als Hilfsausgabe + Neugewichtung nach Uebersehenen + faltenweise physikgestuetzte Synthese positiver CMB und markierter harter Negative; VALDO Task 2
  Laesions-F1 74.3 % ("best local-comparison", p = 0.020), extern AIBL-SWI Recall 88.5 %, F1 65.0 %. Zuordnungsregel und Aufteilung dort noch nicht nachgelesen -- vor einem
  Zahlenvergleich klaeren. Fuer uns neu: Zentrumskarten (verwandt mit dem blob-Verlust), Neugewichtung nach Uebersehenen, Synthese auch der NEGATIVE.
  **Volltext gelesen (arXiv-HTML, 20.09. 19:20):** alle 72 VALDO-Faelle, 5-fach-CV geschichtet nach CMB ja/nein, jeder Fall einmal out-of-fold (NICHT der verdeckte Testsatz);
  Treffer = vorhergesagter Schwerpunkt <= 5 mm vom Referenz-Schwerpunkt, eins-zu-eins (gierig); F1 JE PROBAND, dann gemittelt. Eingabe T1 + T2 + T2* bei 1 mm isotrop, Perzentil-
  Skala 0.5-99.5 + z-Norm, Ausschnitte 96^3 im Verhaeltnis 2:1, 200 Epochen, Schwelle 0.5, < 5 Voxel entfernt. Tabelle I: CenSynCMB R 75.3 / P 79.3 / F1 74.3, 0.8 FP je Proband;
  Attention-U-Net (ihr Rueckgrat) 65.8; DynUNet 61.3; "FRST-style" 63.3; VALDO-Team Zihao 61.1; SwinUNETR 54.3. Ablation: Rueckgrat 65.8 -> + Neugewichtung 67.6 -> + Zentrumskarte
  70.3 -> beides 71.7 -> + synthetische CMB 73.1 -> + synthetische Verwechsler 74.3. Zentrumskarte: Gauss je Blutung (sigma 2 Voxel), fokaler MSE, Gewicht ueber gelernte
  Unsicherheit. Neugewichtung: Gewicht je Trainingsausschnitt aus Blutungsgroesse und aktueller Netzantwort (kleine, schwach erkannte Blutungen zaehlen mehr). Synthese: Dipolkern im
  k-Raum fuer das Blooming, Verwechsler als kalkartige und gefaessartige Quellen, streng faltenweise. Kein Code veroeffentlicht. **Folgerung:** der groesste Einzelbeitrag (+4.5
  Punkte) ist die ZENTRUMSKARTE als Hilfsausgabe -- billig nachzubauen; Synthese bringt +2.6 bei deutlich mehr Aufwand. Unsere Zahlen sind mit ihren erst vergleichbar, wenn wir
  dieselbe Regel rechnen (Schwerpunkt 5 mm, F1 je Proband gemittelt) -- wird nachgeholt.
  **Nachgeholt (20.09. 19:30, `cmb/analysis/subject_level_f1.py`, neu; unsere 57 CV-Faelle, nur T2*, feste Nachbearbeitung + Liquor-Regel):** F1 je Proband gemittelt --
  Startwert 42: 0.571 (nur Probanden mit Blutung) bzw. 0.626 (alle; "nichts da, nichts gemeldet" = 1), Median 0.667, R 0.64 / P 0.62, 1.02 FP je Proband, 14 von 19 blutungsfreien
  Probanden ohne Fehlalarm. Startwert 1: 0.463 / 0.519. Ensemble aus zwei: 0.544 / 0.609 (P 0.64, 0.91 FP). blob: 0.486 / 0.517 (R 0.71, P 0.53, 1.61 FP). Attention-U-Net aus dem
  Turnier (40 Ep., mit Uebermalung): 0.257. **Einordnung:** wir liegen im Bereich IHRER Vergleichsverfahren (61-66), klar unter CenSynCMB (74.3; P 79, 0.8 FP je Proband). Der
  Abstand steckt in der PRECISION -- das Mass je Proband bestraft jeden Fehlalarm bei Probanden mit 1-2 Blutungen hart, und genau das ist unsere schwaechste Klasse. Nicht eins zu
  eins vergleichbar: sie nutzen alle 72 Faelle und drei Sequenzen (T1, T2, T2*), wir 57 Faelle und nur T2*; wie blutungsfreie Probanden in ihr Mittel eingehen, steht nicht im Text.
- *Spiegel-TTA* (`cmb/analysis/tta_predict.py`, neu; acht Spiegelbilder, Mittel der Wahrscheinlichkeiten, weiterhin out-of-fold; CPU-Pruefung: Spiegelungen werden exakt
  zurueckgenommen): eingereiht fuer VALDO-Basis, VALDO-blob und catalina-T2* nur-catalina (je ~25-45 min GPU), Auswertung wie die Ensembles.
**Naechste Schritte:** 3D-ResNet-Stufe auf den blob-Kandidaten (eingereiht), blob + Stufe 2 auf catalina-T2* (5-6x so viele Kandidaten; `candidates.py`
kann seit heute Manifeste), Ensemble ueber Startwerte, danach ggf. Blob-Gewicht 0.25 und harte Negative.

**blob auf catalina-T2* (20.09. 20:40; 75 Faelle, 828 Blutungen; evaluate, missed_lesions, operating_point, stage2):** feste Zelle F1 0.634 gegen 0.663, gepaart
**-0.029 [-0.059; -0.007] -- gesichert schlechter**; Sensitivitaet 0.66 -> 0.70, Precision 0.67 -> 0.58, Fehlalarme je Fall 3.6 -> 5.7 (blutungsfreie Faelle 1.7 -> 3.4). Mit Liquor-Regel:
feste Zelle 0.644 gegen 0.675, kreuzweise 0.642 gegen 0.675, gepaart -0.032 [-0.060; -0.003]. **Nach Groesse (nur-catalina | blob): klein < 10 mm3 (n=239) 0.27 | 0.28, mittel (n=298)
0.72 | 0.81, gross (n=291) 0.91 | 0.95;** Kontrast unter Median 0.46 | 0.52, darueber 0.86 | 0.90. Anders als auf VALDO (klein 0.52 -> 0.64) erreicht der Verlust die KLEINEN catalina-Blutungen
nicht: bei 5-mm-Schichten sind sie ~2 native Voxel, die Information fehlt im Bild -- der Verlust kann nur hervorholen, was da ist. Gewonnen werden mittlere und grosse, bezahlt mit
Fehlalarmen. Nach Ort steigt die Sensitivitaet ueberall leicht (lobaer 0.71 -> 0.76, infratentoriell 0.46 -> 0.52). **Zweite Stufe auf den blob-Kandidaten:** 1135 Kandidaten (605 positiv,
530 falsch), Obergrenze 0.72 (Basis 0.69); nur Schwelle 0.651 | logistisch 0.653 | Wald 0.645 | Boosting 0.629 -- auch mit 5-6x so vielen Beispielen kein Gewinn, und die Kette blob + Stufe 2
(0.653) bleibt UNTER der Basis mit reiner Schwelle (0.670). **Entscheidung:** blob mit Gewicht 0.5 ist fuer catalina-T2* keine Verbesserung; der Nutzen auf VALDO (kleine Blutungen) bleibt
ein Befund fuer duenne Schichten -- naechster sinnvoller Pruefort ist catalina-SWI (1.9 mm). Fehler im Werkzeug behoben: `missed_lesions.py` schrieb abgeschnittene JSON-Dateien (numpy-Zahl
nicht serialisierbar, Schluessel `cases` doppelt belegt); gedruckte Tabellen waren korrekt, alle drei JSON neu erzeugt.
**Spiegel-TTA auf VALDO (20.09. 20:40, acht Ansichten, weiterhin out-of-fold):** Basis 0.585 -> 0.604, gepaart +0.019 [-0.017; +0.058], Fehlalarme je Fall 1.51 -> 1.33 bei gleicher
Sensitivitaet; blob 0.540 -> 0.564, gepaart gegen blob **+0.024 [+0.001; +0.048]**, Fehlalarme 2.30 -> 2.07. Mit Liquor-Regel feste Zelle 0.646 -> 0.652 bzw. 0.599 -> 0.613. Im Mass je
Proband kein Gewinn (0.626 -> 0.621). Kleiner, kostenloser, in beiden Modellen gleichgerichteter Effekt.
**Spiegel-TTA auf catalina-T2* (20.09. 20:50):** KEIN Effekt -- 0.663 -> 0.663, gepaart +0.000 [-0.012; +0.014]; mit Liquor-Regel 0.675 -> 0.671; Fehlalarme je Fall 3.63 -> 3.48. Damit gilt:
TTA hilft auf VALDO (+0.02), auf catalina nicht; sie schadet nirgends und bleibt als Option der Inferenz, ist aber kein Hebel fuer die Fehlalarme auf catalina.
**3D-ResNet als zweite Stufe auf den VALDO-blob-Kandidaten (20.09. 20:45, GPU, geschachtelt, 3 Startwerte gemittelt):** 0.624 (P 0.66, S 0.59, 0.74 FP je Fall) -- gleichauf mit der
logistischen Regression (0.623), +0.016 ueber der reinen Schwelle (0.608), gleichauf mit der Basis-Kette. Auch das CNN sieht also nichts Entscheidendes, was Stufe 1 nicht schon nutzt.

**Ein gemeinsamer Klassifikator fuer alle Kohorten (Nutzeridee 21.09.: "die U-Nets duerfen je Kohorte verschieden sein, der Klassifikator sollte immer dasselbe sehen"; `cmb/stage2/pooled.py`, neu).**
Jede Kohorte behaelt ihre erste Stufe; die Kandidaten aller Kohorten (VALDO 203, catalina-T2* 961, catalina-SWI 870; zusammen 2034, davon 1149 auf einer Blutung) werden vereinigt, EIN Klassifikator
lernt auf allen. Geschachtelte Auswertung wie in 5d (Klassifikator fuer Falte k sieht keine Kandidaten der Falte k, gleich welcher Kohorte; Annahmeschwelle je Kohorte innen gewaehlt; die sieben
SWI-Zwillinge von T2*-Probanden liegen in der Falte ihres Zwillings). Das Skript reproduziert die frueheren Einzelwerte exakt (VALDO 0.627 / 0.616 / 0.579, T2* 0.670 / 0.658).

| Verfahren (gepooltes F1, geschachtelt) | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| nur Schwelle auf die Stufe-1-Wahrscheinlichkeit | 0.627 | 0.670 | 0.626 |
| eigener Klassifikator je Kohorte (logistisch / Wald) | 0.616 / 0.579 | 0.658 / 0.657 | 0.648 / 0.631 |
| GEMEINSAMER Klassifikator (logistisch / Wald) | 0.608 / 0.612 | 0.670 / 0.664 | 0.652 / 0.653 |
| gemeinsam + Kohorten-Kennung (logistisch) | 0.625 | 0.665 | 0.654 |

Gepaart gegen die reine Schwelle (Bootstrap ueber Faelle): **SWI gemeinsam logistisch +0.026 [+0.013; +0.041], mit Kennung +0.028 [+0.013; +0.043], Wald +0.027 [+0.008; +0.052] -- gesichert**; eigener
logistischer +0.022 [+0.005; +0.043]. T2*: gemeinsam -0.000 [-0.012; +0.011]. VALDO: gemeinsam -0.020 [-0.051; +0.008]; der eigene Wald ist dort gesichert schlechter (-0.048), der gemeinsame nicht mehr.
**Gemessen:** (1) die Idee traegt in der Richtung -- der gemeinsame Klassifikator ist in 5 von 6 Vergleichen mindestens so gut wie der eigene, am deutlichsten beim Wald auf der kleinsten Kohorte
(VALDO 0.579 -> 0.612) und auf SWI (0.631 -> 0.653); (2) erstmals schlaegt eine zweite Stufe die reine Schwelle GESICHERT -- aber nur auf SWI. **Vorbehalt vor der Deutung:** SWI ist die einzige Kohorte
ohne Liquor-Regel im Kandidatenfilter (keine Gewebekarten) -- der Klassifikator koennte dort nur nachholen, was die Regel in den anderen Kohorten schon tut. **Gegenprobe laeuft:** SynthSeg `--robust` auf
allen 61 SWI-Faellen (CPU, `mb-arena/synthseg_swi.sh`, dieselben Einstellungen wie T2*), danach SWI-Kandidaten MIT Liquor-Regel und derselbe Vergleich.
**Gegenprobe und Ausbau (21.09. 08:50).** SynthSeg `--robust` lief auf allen SWI-Faellen (62/62, 0 Fehler). Befund: die HARTE Liquor-Regel passt nicht auf SWI -- die Karte legt 74 der 652
Referenzblutungen in den Liquor, 65 davon findet das Netz; die Regel wuerde sie loeschen (F1 0.634 -> 0.608). Der SWI-Gewinn des Klassifikators ist also KEIN Nachholen der Regel. Konsequenz:
Kandidaten werden jetzt OHNE harte Regel gebaut und tragen die Gewebeklasse als Merkmal (`candidates.py --keep-csf`, Feld `tissue`); der gemeinsame Klassifikator bekommt sie als fuenf One-hot-Merkmale
(`pooled.py --tissue`) und lernt je Kohorte selbst, wie viel "liegt im Liquor" zaehlen soll. Dazu (Nutzerfreigabe): Kandidaten der empfindlicheren Netze (blob, Zentrumskarte) als NUR-Trainings-
beispiele (`--extra`; bewertet werden weiter die Kandidaten der Basis-Netze; fuer Falte k nur Zusatzkandidaten von Faellen ausserhalb der Falte).

| Verfahren (geschachtelt, gepooltes F1) | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| nur Schwelle, ohne Regel | 0.565 | 0.654 | **0.626** |
| nur Schwelle + harte Liquor-Regel (bisherige Kette) | **0.627** | **0.670** | 0.614 |
| gemeinsamer Klassifikator ohne Gewebemerkmal (logistisch / Wald) | 0.574 / 0.556 | 0.640 / 0.652 | 0.649 / 0.646 |
| gemeinsam + Gewebemerkmal (logistisch / Wald) | 0.574 / 0.624 | 0.668 / 0.679 | 0.651 / 0.650 |
| gemeinsam + Gewebemerkmal + Zusatzkandidaten (logistisch / Wald) | 0.587 / **0.618** | 0.677 / **0.681** | **0.664** / **0.659** |

Gepaart gegen die JEWEILS beste einfache Kette der Kohorte (VALDO und T2*: Schwelle + harte Regel; SWI: Schwelle ohne Regel), Wald mit Gewebemerkmal und Zusatzkandidaten: VALDO -0.009 [-0.033; +0.019],
T2* +0.011 [-0.009; +0.036], **SWI +0.033 [+0.013; +0.057]** (logistisch auf SWI +0.039 [+0.017; +0.070]; Precision 0.65 -> 0.74, Fehlalarme je Fall 3.5 -> 2.2 bei gleicher Sensitivitaet).
**Gemessen:** EIN Klassifikator fuer alle Kohorten erreicht auf VALDO und T2* die harte Regel (er lernt sie aus dem Merkmal) und ist auf SWI gesichert besser als alles Bisherige; die Zusatzkandidaten bringen
je Kohorte weitere +0.002 bis +0.014. **Interpretiert:** die zweite Stufe scheiterte bisher nicht an der Idee, sondern daran, dass ihre Merkmale nichts Neues enthielten -- die Gewebeklasse IST neu, und ein
gemeinsamer Pool gibt genug Beispiele, sie kohortengerecht zu gewichten. **Grenzen:** (1) viele Varianten an denselben Faellen verglichen -- der SWI-Gewinn ist ueber alle zehn Varianten gleichgerichtet
(+0.018 bis +0.039), die Wahl der besten Variante aber nachtraeglich; als Endkonfiguration wird VORAB festgelegt: Wald, Gewebemerkmal, Zusatzkandidaten. (2) Bestaetigung mit unabhaengiger erster Stufe
steht aus (SWI-Basis mit Startwert 1 eingereiht). (3) Das gemeinsame 3D-ResNet (GPU) ist eingereiht.
**Empfindliche erste Stufe + gemeinsames 3D-CNN (21.09. 14:15, GPU ~80 min; blob-Kandidaten bewertet, Basis- und Zentrumskarten-Kandidaten als Zusatztraining):** VALDO 0.594 / mit Zusatz 0.607, catalina-T2* 0.673 / 0.682,
catalina-SWI 0.677 / 0.671. Gepaart ueber dieselben Faelle gegen die BASIS-Stufe mit demselben 3D-CNN (`cmb/stage2/compare_runs.py`): SWI +0.007 [-0.024; +0.034], T2* +0.000 [-0.016; +0.022], VALDO -0.036 [-0.117; +0.041].
Gegen die bisherige einfache Kette: SWI +0.051 [+0.013; +0.092], T2* +0.012 [-0.009; +0.040], VALDO -0.020 [-0.074; +0.046]. **Entscheidung:** die empfindliche erste Stufe zahlt sich auch mit dem staerksten Klassifikator nicht
aus -- der Gewinn kommt von der zweiten Stufe, nicht vom laesionsweisen Verlust. Endkette damit: Basisrezept-U-Net je Kohorte -> gemeinsames 3D-CNN (mit Zusatzkandidaten im Training, das hilft T2*).
**BESTAETIGUNGSVERSUCH mit unabhaengiger SWI-Stufe (Startwert 1; 21.09. 16:15) -- die vorab festgelegte Endkonfiguration bestaetigt sich NICHT.** Erste Stufe Startwert 1 auf SWI: feste Zelle 0.613 gegen 0.634
(gepaart -0.022 [-0.065; +0.012]); 893 weiche Kandidaten, Obergrenze 0.68. Gemeinsamer Wald + Gewebemerkmal + Zusatzkandidaten: SWI 0.621 gegen reine Schwelle 0.617, gepaart **+0.004 [-0.021; +0.041]**; logistisch -0.001
[-0.025; +0.031]. Die Precision steigt wie zuvor (0.61 -> 0.68), aber die Sensitivitaet faellt (0.62 -> 0.57) -- unterm Strich nichts. T2* unveraendert +0.010 / +0.026 gegen Regel / Schwelle. Nebenbefund: VALDO faellt in diesem Lauf auf
0.579 (vorher 0.618), obwohl sich NUR die SWI-Kandidaten im Pool geaendert haben -- der VALDO-Wert des gemeinsamen Klassifikators schwankt also um +-0.04 mit der Pool-Zusammensetzung (278 Kandidaten, Schwellenwahl an wenigen Faellen).
**Stand der Evidenz:** der SWI-Gewinn der Merkmals-Klassifikatoren (+0.033 / +0.039 mit Startwert 42) ist mit einer unabhaengigen ersten Stufe NICHT reproduziert -- er gilt damit als nicht bestaetigt. Offen ist das 3D-CNN (groesster Gewinn,
+0.044): Bestaetigungslauf auf den Startwert-1-Kandidaten eingereiht (GPU, heute Abend). Bis dahin gilt als Endkette weiter die einfache: U-Net + feste Nachbearbeitung + harte Liquor-Regel (VALDO, T2*) bzw. ohne Regel (SWI).
**BESTAETIGUNG des 3D-CNN mit unabhaengiger SWI-Stufe (21.09. 20:00, GPU ~30 min):** auf den Kandidaten des Startwert-1-Netzes erreicht das gemeinsame 3D-CNN SWI 0.662 (P 0.74, S 0.60, 2.25 Fehlalarme je Fall) gegen
reine Schwelle 0.617: gepaart **+0.045 [+0.018; +0.085]** -- praktisch derselbe Gewinn wie mit Startwert 42 (+0.044 [+0.012; +0.085]). VALDO 0.636 (gegen die harte-Regel-Kette +0.009 [-0.031; +0.054]), T2* 0.672 (+0.001).
**Stand der Evidenz, berichtigt:** der SWI-Gewinn des 3D-CNN ist mit zwei unabhaengig trainierten ersten Stufen reproduziert; der Gewinn der MERKMALS-Klassifikatoren (Wald, logistisch) nicht. Das passt zur Deutung aus 5n: das CNN
sieht das Bild um den Kandidaten (Gefaessverlauf, Sulcus, Liquor) und lernt daran, was KEINE Blutung ist; zehn Zahlen je Kandidat reichen dafuer nicht. **Entscheidung:** Endkette = Basis-U-Net je Kohorte -> gemeinsames 3D-CNN;
auf VALDO und T2* ersetzt es die harte Liquor-Regel gleichwertig (ohne Gewebekarte zu brauchen), auf SWI bringt es +0.045 und rund zwei Fehlalarme weniger je Fall. Noch offen: dieselbe Bestaetigung fuer VALDO und T2* mit den dortigen
zweiten Startwerten (VALDO: Kandidaten liegen vor, T2*: kein zweiter Startwert vorhanden) und die Schlussmessung auf den 15 beiseitegelegten VALDO-Faellen.
**Berichtigung der Bezeichnung (21.09. 14:40, Nutzerfrage nach der Architektur):** das in 5d und hier "3D-ResNet" genannte Netz ist KEIN ResNet -- es hat keine Residualverbindungen. Es ist ein kleines, schlichtes 3D-CNN:
Eingabe 3 Kanaele (T2* invertiert, FRST, Wahrscheinlichkeit der ersten Stufe), Ausschnitt 16 x 32 x 32 Voxel = 16 x 16 x 16 mm um den Kandidaten; drei Doppelfaltungs-Bloecke (3x3x3, InstanceNorm, LeakyReLU) mit 16 / 32 / 64
Kanaelen, dazwischen zweimal MaxPool 2, dann globales Mittel, Dropout 0.3, eine lineare Ausgabe; 216 081 Parameter. Training: 40 Epochen, AdamW 1e-3 (Gewichtsabfall 1e-3), Kosinus, Stapel 16, BCE mit Klassengewicht
n_neg / n_pos, Augmentierung durch Zufallsversatz (+-2 Schichten, +-4 Voxel in der Schicht) und Spiegelungen; drei Startwerte gemittelt. In Text und Tabellen ab jetzt "3D-CNN"; aeltere Eintraege bleiben unveraendert stehen.
**Gemeinsames 3D-ResNet (21.09. 10:20, GPU ~25 min; `pooled.py --models cnn`, weiche Kandidaten, OHNE Gewebemerkmal und ohne Zusatzkandidaten):** VALDO 0.625 (P 0.65, S 0.60, 0.81 Fehlalarme je Fall),
catalina-T2* 0.672 (P 0.70, 3.0), **catalina-SWI 0.670** (P 0.74, S 0.61, 2.3). Gepaart gegen die beste einfache Kette je Kohorte: VALDO -0.003 [-0.046; +0.038], T2* +0.001 [-0.014; +0.022],
**SWI +0.044 [+0.012; +0.085]**. Das Netz sieht keine Gewebekarte und erreicht auf VALDO und T2* trotzdem die harte Liquor-Regel -- es erkennt den Liquor am Bild selbst; auf SWI ist es das bisher beste
Ergebnis (0.626 -> 0.670). Zum Vergleich: das kohorteneigene 3D-ResNet auf VALDO lag bei 0.599 (5d) -- der gemeinsame Pool hilft auch dem CNN. Eingereiht: dasselbe mit den Zusatzkandidaten der
empfindlichen Netze (`--extra`).
**Zwei Gegenproben auf der CPU (21.09. 11:10), bevor daraus ein Planwechsel wird:** (a) EMPFINDLICHE erste Stufe (blob) als BEWERTETE Kandidaten, gemeinsamer Wald + Gewebemerkmal + Zusatztraining: VALDO 0.578,
T2* 0.671, SWI 0.655 -- nicht besser als die Basis-Stufe mit demselben Klassifikator (0.618 / 0.681 / 0.659); die hoehere Obergrenze der empfindlichen Stufe (0.79 / 0.74 / 0.73) setzt der Wald nicht um. Ob das
3D-ResNet es kann, prueft ein eingereihter GPU-Lauf. (b) VALDO mit UNABHAENGIGER erster Stufe (Startwert 1): harte-Regel-Kette 0.559, gemeinsamer Wald 0.531 (-0.029 [-0.061; +0.011]), mit Zusatztraining 0.507
(-0.053 [-0.110; +0.004]); T2* und SWI unveraendert (+0.009 / +0.045 gesichert auf SWI). **Stand:** der Gewinn der zweiten Stufe ist auf SWI robust (drei Klassifikatorfamilien, zwei Pool-Zusammensetzungen), auf T2* klein
und ungesichert, auf VALDO nicht vorhanden -- dort bleibt die harte Liquor-Regel (T1-basierte, genaue Karte) mindestens gleichwertig. Werkzeug: `cmb/stage2/compare_runs.py` fuer gepaarte Vergleiche ueber
Kandidatensaetze hinweg (pooled.py schreibt jetzt Zaehlungen je Fall).
**3D-ResNet mit Zusatzkandidaten (21.09. 12:50, GPU ~55 min):** VALDO 0.622, catalina-T2* **0.684** (P 0.72, S 0.65, 2.7 Fehlalarme je Fall; gegen die Kette +0.014 [-0.002; +0.035]), SWI 0.669 (+0.044 gegen die
Schwelle, gesichert). Der Wiederholungslauf OHNE Zusatzkandidaten im selben Auftrag ergab 0.630 / 0.673 / 0.670 (erster Lauf 0.625 / 0.672 / 0.670) -- die GPU-Streuung des CNN liegt also bei ~0.005. Die Zusatzkandidaten
helfen nur T2* (+0.011), sonst nichts. **Planwechsel (Nutzerfreigabe 21.09. 11:15):** weitere GPU-Zeit geht an die zweite Stufe und ihre Bestaetigung; gestrichen sind sieben Eintraege mit U-Net-Varianten ohne Aussicht:
Verhaeltnis 1:1 + Kette (1:2 kollabiert, 5g), Attention auf dem Sieger, Perzentil-Skala Startwert 1, Feintuning 60 Ep. (T2*, SWI), Gewebekanal auf catalina (VALDO -0.058, 5m). Behalten: 3D-ResNet auf den Kandidaten der
empfindlichen ersten Stufe (laeuft), SWI-Netz Startwert 1 (unabhaengige Bestaetigung), gemischt / eigen mit 120 Ep. (Budget-Frage), kleiner und grosser Ausschnitt (Kontext), Basis Startwert 2.

### 5k Zentrumskarte als Hilfsausgabe (nach CenSynCMB 2026), eingereiht 20.09. abends
**Original / Literatur:** MicrobleedNet hat keine instanzbezogene Aufsicht; CenSynCMB (He et al., arXiv 2607.05325) fuehrt eine Zentrumskarte als zweiten Ausgabekanal ein --
groesster Einzelbeitrag ihrer Ablation (65.8 -> 70.3 F1 je Proband). **Beobachtung bei uns:** unser Abstand zu CenSynCMB steckt in der Precision (0.62 gegen 0.79); der
laesionsweise Verlust (5j) hebt die Sensitivitaet kleiner Blutungen, aber auch die Fehlalarme. **Hypothese:** ein Kopf, der je Blutung EINEN Gauss-Hoecker am Schwerpunkt lernen
muss, zwingt die gemeinsamen Merkmale zu "hier ist genau eine kompakte Laesion" statt "hier ist es dunkel" -- das sollte Fehlalarme an Gefaessen und Raendern eher senken als heben.
**Aenderung (`code/turnier.py`, Sicherung turnier.py.vor-zentrum-0920):** Netz `a12-anisozentrum` = a03-aniso + 1x1x1-Kopf auf denselben Decoder-Merkmalen; im Training zwei
Kanaele, in eval() nur die Segmentierung (alle Werkzeuge unveraendert nutzbar; Gewichte = a03-Schluessel + `netz.zentrum.*`). Verlust `--verlust zentrum` = BCE+Dice + fokaler MSE
auf der Zentrumskarte, Gewicht (C + eps)^2.5. **Bewusste Abweichungen vom Paper:** sigma 2 mm in mm statt 2 Voxel (unser Gitter ist 0.5 x 0.5 x 1 mm); genormt auf die
Gewichtssumme mit festem Gewicht 1 statt gelernter Unsicherheit (unser Verlust sieht die Netzparameter nicht); eps 0.05 statt 1e-3, damit der Kopf auch "kein Zentrum" im
Hintergrund lernt (spaeter als zweite Meinung fuer die zweite Stufe nutzbar). **Pruefung (CPU):** Hoecker sitzt am Schwerpunkt, 2 mm entfernt 0.53-0.60 in x wie in z; leerer
Ausschnitt -> Karte 0; Gradient erreicht Kopf und gemeinsame Merkmale; eval() liefert einen Kanal; Speichern/Laden bitgleich; Schalter und Netz nur gemeinsam zulaessig.
**Experiment:** 60 Ep., gitter_d, Startwert 42, sonst Basisrezept; Vergleich gepaart gegen die Basis an der festen Zelle, dazu Mass je Proband (5j) und Fehlalarme nach Ort.
Noch NICHT nachgebaut: Neugewichtung nach Uebersehenen (+1.8 bei ihnen), Synthese (+2.6).
**Ergebnis (20.09. 22:55, 5/5 Falten ohne Fehler, Laufzeit je Falte wie die Basis):** feste Zelle F1 0.531 gegen 0.585, gepaart **-0.054 [-0.110; -0.006] -- gesichert schlechter**;
Sensitivitaet 0.67 -> 0.73, Precision 0.52 -> 0.42, Fehlalarme je Fall 1.51 -> 2.51 (86 -> 143; davon Liquor 52, lobaer 58, tief 20). Mit Liquor-Regel: feste Zelle 0.606 gegen 0.646, beste
Einzelzelle 0.6 / 4 mm3 = 0.636 (optimistisch), kreuzweise 0.586 gegen 0.622 (-0.036 [-0.089; +0.020]). Mass je Proband (CenSynCMB-Protokoll): 0.531 gegen 0.571 (Probanden mit Blutung),
0.459 gegen 0.626 ueber alle -- nur 6 statt 14 der 19 blutungsfreien Probanden bleiben ohne Fehlalarm. Sensitivitaet nach Groesse (Basis | Zentrum | blob): klein 0.52 | 0.66 | 0.64, mittel
0.69 | 0.69 | 0.73, gross 0.79 | 0.85 | 0.79; kontrastarm 0.51 | 0.59 | 0.55.
**Hypothese widerlegt:** die Zentrumskarte senkt unsere Fehlalarme NICHT -- sie wirkt wie der laesionsweise Verlust: empfindlicher (kleine Blutungen +14 Punkte), anders kalibriert
(Optimum bei Schwelle 0.6 statt 0.3), mehr Fehlalarme, am eigenen besten Arbeitspunkt bestenfalls gleichauf. Der Gewinn aus dem Paper (+4.5) uebertraegt sich auf unseren Aufbau nicht.
**Moegliche Gruende (nicht geprueft):** (1) sie werten bei Schwelle 0.5 mit Checkpoint-Wahl nach Validierungs-F1 aus, wir an der festen Zelle 0.3, die fuer instanzbewusste Modelle unguenstig
liegt; (2) mit T1 + T2 lassen sich die zusaetzlich gefundenen Verwechsler abweisen, mit T2* allein nicht -- mehr Sensitivitaet zieht dann Gefaesse und Liquor-Artefakte mit; (3) meine
Abweichungen (Hintergrundterm eps 0.05, festes Gewicht); (4) 60 statt 200 Epochen. **Entscheidung:** negatives Ergebnis festhalten, KEIN zweiter Startwert, keine Kombination mit fnrw.
**Zweite Stufe auf den Zentrumskarten-Kandidaten (Nutzerfrage 20.09. 23:40 -- die Zahlen oben sind NUR die erste Stufe):** 275 Kandidaten (117 positiv, 158 falsch), Obergrenze 0.77;
geschachtelt: alle annehmen 0.530 | nur Schwelle auf die eigene Wahrscheinlichkeit **0.635** (P 0.70, S 0.58, 0.61 FP je Fall) | logistisch 0.595 | Wald 0.606. Der zweite Klassifikator
hilft auch hier nicht; eine hoehere, geschachtelt gewaehlte Schwelle bringt das Modell auf das Niveau der Basis-Kette (0.627), nicht darueber.
**Uebergreifender Befund aus 5j + 5k:** beide Hebel, die das Netz instanzbewusster machen, finden mehr kleine Blutungen und bezahlen mit Fehlalarmen; das F1 aendert sich nicht zum Guten.
Unser Engpass ist damit nicht "zu wenig Sensitivitaet im Training", sondern die TRENNUNG von Blutung und Verwechsler bei nur einer Sequenz. Folge: harte Negative (setzt an den Fehlalarmen an)
vor fnrw gezogen (fnrw betont wieder die Uebersehenen und duerfte dasselbe Muster zeigen); naechste sinnvolle Kombination waere blob/Zentrum + harte Negative.

### 5l Zwei weitere Trainingshebel gegen Fehlalarme und Uebersehene (eingebaut 20.09. abends, Nutzerfreigabe)
**(a) Neugewichtung nach Uebersehenen, `--verlust fnrw`** (CenSynCMB, dort +1.8 F1-Punkte allein, +5.9 zusammen mit der Zentrumskarte): BCE+Dice je Ausschnitt, gewichtet mit
w = (mittlere Groesse / Groesse)^0.5 * (1 + 1.5 * (1 - hoechste aktuelle Netzantwort auf der Blutung)), begrenzt auf [0.5, 5]; Ausschnitte ohne Blutung Gewicht 1. Das Paper sagt nicht,
wie mehrere Blutungen in einem Ausschnitt zusammengehen -- hier das Maximum. Unterschied zum laesionsweisen Verlust (5j): dieser hier gewichtet nach dem AKTUELLEN Koennen des Netzes
(gefundene Blutungen verlieren Gewicht), blob gewichtet starr nach Instanz. CPU-Pruefung: ohne Blutungen exakt cv5.loss_bce_dice; kleine uebersehene Blutung Gewicht 1.06 -> 1.00, sobald
gefunden; Riesenlaesion 0.5; Gradient endlich.
**(b) Harte Negative, `--harte-negative`** (klassisch, fuer CMB Dou et al. 2016): ab Epoche 11 sagt alle 10 Epochen das AKTUELLE Netz seine eigenen Trainingsfaelle vorher (feste
Nachbearbeitung); Schwerpunkte der Fehlalarm-Komponenten werden Zentren fuer 60 % der blutungsfreien Ausschnitte. Streng innerhalb der Falte (kein fremder Lauf, kein Testfall), das Label
im Ausschnitt bleibt das echte. Kosten: fuenf Schuerfrunden je Falte (geschaetzt +8 min). CPU-Pruefung mit kuenstlichem Fehlalarm: vor dem Schuerfen 1 von 22 blutungsfreien Ausschnitten am
Fehlalarm, danach 13 von 21; Netz danach wieder im Trainingsmodus; Zustand wird je Falte zurueckgesetzt.
**Experimente (eingereiht hinter der Zentrumskarte):** blob auf catalina-SWI (1.9-mm-Schichten -- erreicht der Verlust dort die kleinen Blutungen wie auf VALDO?), fnrw auf VALDO,
harte Negative auf VALDO; jeweils 60 Ep., Startwert 42, gepaart gegen die Basis an der festen Zelle, dazu Fehlalarme je Fall und Sensitivitaet nach Groesse. Gestrichen: blob Startwert 2.
**Ergebnisse (21.09. 05:35, alle Laeufe 5/5 Falten ohne Fehler; feste Zelle, gepaart gegen die Basis mit Startwert 42):**

| Lauf | F1 | Precision | Sensitivitaet | Fehlalarme je Fall | gepaart | mit Liquor-Regel (fest) | F1 je Proband | klein / mittel / gross |
|---|---|---|---|---|---|---|---|---|
| Basis | 0.585 | 0.52 | 0.67 | 1.51 | Bezug | 0.646 | 0.571 | 0.52 / 0.69 / 0.79 |
| harte Negative | 0.585 | 0.53 | 0.66 | 1.42 | +0.000 [-0.044; +0.048] | 0.625 | 0.546 | 0.52 / 0.67 / 0.77 |
| Neugewichtung (fnrw) | 0.555 | 0.46 | 0.69 | 1.95 | -0.030 [-0.080; +0.019] | 0.592 | 0.506 | 0.55 / 0.71 / 0.81 |

**Harte Negative: kein Effekt.** Das Schuerfen selbst arbeitet (Falte 4: 86 Fehlalarm-Zentren in Epoche 31, 112 in Epoche 41, 42 in Epoche 51 -- auf den TRAININGSfaellen lernt das Netz
seine Fehlalarme also weg), aber auf den Testfaellen aendert sich nichts. **Deutung (Hypothese):** die Fehlalarme sind fallspezifisch und werden auswendig gelernt, nicht als Klasse
verallgemeinert -- bei 33 Trainingsfaellen je Falte gibt es zu wenige Beispiele je Verwechsler-Art. **fnrw:** dasselbe Muster wie blob und Zentrumskarte, nur schwaecher (mehr
Sensitivitaet, mehr Fehlalarme, F1 nicht besser). **Entscheidung:** beide nicht uebernommen; keine Kombinationen.
**blob auf catalina-SWI (61 Faelle, 652 CMB; nur Summen):** F1 0.631 gegen 0.634, gepaart -0.003 [-0.036; +0.024] -- gleichauf; Sensitivitaet 0.65 -> 0.68, Fehlalarme je Fall 4.3 -> 5.0.
Nach Groesse: klein < 9.9 mm3 (n=217) **0.37 -> 0.44**, mittel 0.70 -> 0.73, gross 0.88 -> 0.86. Die Hypothese aus 5j haelt: bei duennen Schichten (SWI 1.9 mm, VALDO) erreicht der Verlust
die kleinen Blutungen, bei 5-mm-T2* nicht -- das F1 verbessert er nirgends. (Werkzeug: `missed_lesions.py` laeuft jetzt auch ohne Gewebekarte; fuer SWI gibt es keine SynthSeg-Karten.)
**Gesamtbilanz der Trainingshebel (5j-5l):** blob, Zentrumskarte, fnrw, harte Negative -- KEINER verbessert das F1; drei machen das Netz empfindlicher und bezahlen mit Fehlalarmen, einer
wirkt nur auf den Trainingsfaellen. Das Basisrezept bleibt. Der Befund stuetzt die Diagnose aus 5k: der Engpass ist die Information im Bild (eine Sequenz, grobe Schichten), nicht das Trainingssignal.

**blob mit zweitem Startwert (21.09. 06:20):** Startwert 1 blob 0.490 gegen Basis Startwert 1 0.553, gepaart -0.063 [-0.129; +0.006] (Sensitivitaet 0.60 -> 0.70, Fehlalarme je Fall 1.42 -> 2.81).
Zwei-Startwerte-Ensembles an der festen Zelle mit Liquor-Regel: Basis 0.647, blob 0.604. Der Befund aus 5j (empfindlicher, mehr Fehlalarme, F1 niedriger) wiederholt sich mit dem zweiten
Startwert in gleicher Richtung und Groesse (-0.044 / -0.063) -- fuer VALDO damit abgeschlossen.

### 5m Gewebekarte als Eingabekanaele (Nutzerfreigabe 21.09. frueh; eingereiht)
**Ausgangslage:** kein Trainingshebel verbessert das F1 (5j-5l); der Engpass ist die Trennung Blutung / Verwechsler mit einer Sequenz. FLAIR scheidet fuer catalina aus (Nutzer 20.09.: zu wenige
Faelle, hilft hoechstens beim Nachbearbeiten). **Beobachtung:** die einzige Massnahme mit "anderer Information", die wir haben, ist die Liquor-Regel aus der SynthSeg-Karte -- und sie ist
unser groesster einzelner Nachbearbeitungsgewinn (VALDO +0.06). **Hypothese:** bekommt das Netz die Karte als EINGABE, lernt es die Regel selbst und darueber hinaus ortsabhaengiges Aussehen
(Sulci, Ventrikelrand, Stammganglien), statt dass wir nachtraeglich eine einzige harte Regel anwenden. **Aenderung (`code/turnier.py --gewebe [MUSTER]`, Sicherung turnier.py.vor-gewebe-0921):**
fuenf One-hot-Kanaele (Liquor, Rinde + mesiotemporal, Mark, tiefe Kerne, Kleinhirn + Hirnstamm) zusaetzlich zu T2* und FRST = 7 Eingabekanaele; die Karte liegt je Fall als EIN uint8-Volumen
im Speicher und wird erst im Ausschnitt aufgefaechert. Stichprobe und Schiebefenster sind wortgleiche Kopien der Originale mit den Zusatzkanaelen. **Pruefung (CPU):** gleiche Zufallsfolge ->
Bild, FRST und Label der Ausschnitte bitgleich mit der Basis-Stichprobe (der Vergleich mit der Basis ist damit auch in den gezogenen Ausschnitten gepaart); One-hot korrekt; Schiebefenster
bitgleich mit der alten Inferenz; Lader auf VALDO (Anteile am Hirn: Liquor 0.20, Rinde 0.31, Mark 0.26, tief 0.03, infratentoriell 0.10) und auf catalina mit Umrechnung der 1-mm-Karte auf das
Gitter (0.18 / 0.34 / 0.27 / 0.02 / 0.10); Netz mit 7 Kanaelen rechnet vorwaerts und rueckwaerts. **Experimente:** VALDO (`--gewebe`, Karte aus T1 im T2*-Raum) und catalina-T2* (Karte aus
SynthSeg `--robust` direkt auf T2*), je 60 Ep., Startwert 42; Auswertung an der festen Zelle MIT und OHNE Liquor-Regel (lernt das Netz die Regel selbst, schrumpft ihr Zusatznutzen), gepaart
gegen die Basis, Fehlalarme nach Ort. **Vorbehalt vorab:** auf VALDO stammt die Karte aus dem T1 -- ein Gewinn dort belegt den Wert von ANATOMIE, nicht den Wert einer T2*-eigenen Karte;
massgeblich fuer die Zielkohorte ist der catalina-Lauf. SWI hat keine Karten (noch nicht gerechnet). Entscheidungsregel wie immer: unter +0.03 nicht vom Startwert-Rauschen zu trennen.
**Ergebnis VALDO (21.09. 08:50, 5/5 Falten, ~28 min je Falte):** feste Zelle F1 0.527 gegen 0.585, gepaart -0.058 [-0.138; +0.013]; Precision 0.52 -> 0.43, Fehlalarme je Fall 1.5 -> 2.2.
Die Hypothese stimmt zur Haelfte: das Netz lernt die Liquor-Regel tatsaechlich selbst -- es macht KEINE EINZIGE Vorhersage im Liquor mehr (0 statt 32 Fehlalarme dort; die Regel hat danach nichts mehr
zu tun, mit Regel 0.527 gegen 0.646 der Basis, gepaart -0.111 [-0.214; -0.007]). Aber ausserhalb des Liquors steigen die Fehlalarme (lobaer 36 -> 81, tief 11 -> 32), und die 5 Referenzblutungen mit
Liquor-Label sind alle verloren. **Interpretiert:** sieben Eingabekanaele bei 57 Faellen -- das Netz stuetzt sich auf die Karte statt auf das Bild. **Entscheidung:** als U-Net-EINGABE nicht uebernommen.
Dieselbe Information wirkt dagegen als MERKMAL der zweiten Stufe (5j, gemeinsamer Klassifikator): Anatomie hilft bei der Entscheidung ueber Kandidaten, nicht als Rohkanal eines kleinen Trainingssatzes.
Der catalina-Lauf bleibt wie angekuendigt eingereiht (Karte dort aus T2* selbst).

### 5n Warum hilft Mischen nicht -- unterscheiden sich die Bilder oder die Experten-Markierungen? (Nutzerfrage 21.09.; `cmb/analysis/cohort_differences.py`, neu)
Gemessen auf dem gemeinsamen Trainingsgitter, je Kohorte (catalina nur als Summen):

| Kohorte | Faelle / CMB | CMB je Fall (Median / Max) | Laesion in nativen Voxeln (Median) | Anteil < 4 native Voxel | Kontrast (rel. Abdunklung, Median) | Hirn-Median (invertierte Skala) | "Verwechsler-Last" |
|---|---|---|---|---|---|---|---|
| VALDO-1 (4 mm) | 8 / 32 | 3 / 12 | 15.5 | 0 % | 0.343 | 0.67 | 2.8 % |
| VALDO-2 (0.8 mm) | 27 / 80 | 0 / 26 | 38.0 | 0 % | 0.384 | 0.72 | 20.9 % |
| VALDO-3 (3-4 mm) | 22 / 27 | 1 / 2 | 5.3 | 41 % | 0.248 | 0.70 | 38.6 % |
| catalina-T2* (5 mm) | 75 / 828 | 3 / 112 | 6.3 | 29 % | 0.212 | 0.53 | 15.3 % |
| catalina-SWI (1.9 mm) | 61 / 652 | 1 / 102 | 21.3 | 5 % | 0.200 | 0.71 | 33.7 % |

(Verwechsler-Last = Anteil der Hirnvoxel, die mindestens so hell sind wie die mittlere Blutung der Kohorte. Ein zusaetzlich gerechneter "Umriss-Index" -- wie viel der Blutungshelligkeit noch im ersten
Voxelring AUSSERHALB der Maske steckt -- liegt bei 0.21-0.32 fuer duenne und 0.41 fuer dicke Schichten; er folgt der Aufloesung, nicht der Kohorte, und taugt deshalb nicht als Mass fuer den Markierstil.)
**Gemessen -- BEIDES unterscheidet sich:** (1) Bilder: Intensitaetsniveau (catalina-T2* 0.53 gegen ~0.70 -- enge HD-BET-Maske, anderes Maximum), natives Voxelvolumen 0.19 bis 3.0 mm3, Verwechsler-Last 3 bis 39 %.
(2) Markierungen: die catalina-Experten haben deutlich SCHWAECHERE Blutungen markiert (Abdunklung 20-21 % gegen 34-38 % in VALDO-1/2) -- und zwar auch auf SWI mit duennen Schichten, also nicht nur ein Effekt der
5-mm-Schichten; dazu Extremfaelle mit ueber 100 Blutungen, die es in VALDO nicht gibt.
**Gegenprobe, ob der Markierstil das Netz stoert:** das NUR auf VALDO trainierte Netz findet auf catalina-T2* dieselben Blutungen wie das catalina-Netz -- nach Groesse 0.29 / 0.71 / 0.87 gegen 0.27 / 0.72 / 0.91,
kontrastschwach 0.48 gegen 0.46, kontraststark 0.82 gegen 0.86. Was es schlechter macht, sind die FEHLALARME: 357 gegen 272 (lobaer 217 gegen 156, Liquor 95 gegen 59). Das gemischte Netz: Sensitivitaet in allen
Klassen hoeher (0.32 / 0.78 / 0.94), aber 6.1 statt 3.6 Fehlalarme je Fall.
**Interpretiert:** Was eine Blutung IST, lernen die Netze kohortenuebergreifend gleich -- der Markierstil ist fuer die Sensitivitaet kein Hindernis. Was KEINE Blutung ist (Gefaessquerschnitte, Sulci, Artefakte),
sieht je Scanner, Sequenz und Schichtdicke anders aus und uebertraegt sich nicht; dort kostet das Mischen. Das spricht staerker fuer BILD- als fuer Experten-Unterschiede. Offen bleibt, wie viele der zusaetzlichen
"Fehlalarme" echte, nicht markierte Blutungen sind -- das kann nur die Expertenpruefung klaeren. Passend dazu: die zweite Stufe laesst sich mischen (5j), weil sie lokal und auf physikalischen Merkmalen bzw.
zentrierten Ausschnitten entscheidet; das U-Net muss das ganze Erscheinungsbild einer Kohorte modellieren.

### 5o Gegenprobe "wirklich feste Epochenzahl" (Nutzerfreigabe 21.09.; eingereiht)
**Ausgangslage (Abschnitt 11):** der Plan laeuft fest 60 Epochen, verwendet wird aber der beste Zwischenstand der inneren Falte (~11 Faelle, alle 5 Epochen) -- gewaehlt wurden z. B. 40 / 60 / 50 / 35 / 60 (Startwert 42) und 50 / 55 / 10 / 40 / 40
(Startwert 1). **Hypothese:** diese Wahl ist selbst eine Rauschquelle; mit dem LETZTEN Zwischenstand (Lernrate am Ende des Kosinus-Plans nahe null, Gewichte "ausgekuehlt") sinkt die Streuung zwischen Startwerten, das mittlere F1 bleibt gleich oder steigt.
**Aenderung:** `cv5.LETZTER_STAND` (Vorgabe False, Verhalten sonst unveraendert; Sicherung cv5.py.vor-letzter-stand-0921) und `turnier.py --letzter-stand` (Ordnerzusatz -ls); Schwelle und Mindestgroesse kommen weiter von der inneren Falte, fuer die
Auswertung zaehlt ohnehin die feste Zelle. **Pruefung (CPU, 2 Epochen, inneres F1 kuenstlich fallend):** gewaehlt wird Epoche 2 (der letzte Stand); die alte Regel haette Epoche 1 genommen. **Experiment:** Basisrezept mit `--letzter-stand` fuer
Startwert 42 und Startwert 1. Da das Training deterministisch ist, unterscheiden sich die Laeufe von den vorhandenen NUR im verwendeten Zwischenstand. **Auswertung (vorab festgelegt):** feste Zelle mit und ohne Liquor-Regel, gepaart gegen den
jeweiligen Lauf mit Zwischenstandswahl; Hauptmass ist der ABSTAND zwischen den beiden Startwerten (bisher 0.585 gegen 0.553 roh, 0.646 gegen 0.590 mit Regel). Wird der Abstand kleiner und das Mittel nicht schlechter, wird der letzte Stand zum Rezept.
**Ergebnis Startwert 42 (21.09. 21:25, 5/5 Falten; `ergebnisse/analysis/letzter_stand_s42_valdo*.json`, `operating_point_valdo_letzter_stand_s42.json`):** Pruefung der Deterministik bestanden -- die Falten 1 und 4, in denen die Basis ohnehin Epoche 60
gewaehlt hatte, sind identisch (0.500 / 0.577). Feste Zelle roh: **0.538 gegen 0.585, gepaart -0.047 [-0.088; -0.010] (gesichert schlechter)**; P 0.44 gegen 0.52, S 0.69 gegen 0.67, Fehlalarme je Fall 2.14 gegen 1.51. Je Kohorte -0.050 / -0.025 /
-0.074 (alle drei negativ, einzeln nicht gesichert). Mit Liquor-Regel an der festen Zelle 0.594 gegen 0.646. **Liegt es nur an der Schwelle?** Nein: der letzte Stand ist selbstsicherer, sein bestes Feld liegt bei Schwelle 0.6 statt 0.3 -- aber auch
dort erreicht er nur 0.618 (optimistisch gewaehlt) gegen 0.646; kreuzweise gewaehlt 0.579 gegen 0.622, gepaart -0.043 [-0.096; +0.011]. **Interpretiert:** die Zwischenstandswahl auf der inneren Falte ist hier KEINE blosse Rauschquelle, sondern
wirkt wie ein Fruehstopp gegen Ueberanpassung (35 Trainingsfaelle): spaete Epochen machen das Netz empfindlicher und selbstsicherer, die Fehlalarme steigen. Bemerkenswert: mit dem letzten Stand landet der "Gluecksfall" Startwert 42 (0.594) genau
bei Startwert 1 mit Zwischenstandswahl (0.590) -- ein Teil seines Vorsprungs steckt also in gut getroffenen Zwischenstaenden. **Entscheidung:** faellt nach der vorab festgelegten Regel erst mit Startwert 1 (laeuft, ~23:05); fuer Startwert 42
ist die Bedingung "Mittel nicht schlechter" bereits verletzt.
**Ergebnis Startwert 1 und Entscheidung (21.09. 23:04; `ergebnisse/analysis/letzter_stand_s1_valdo_gepoolt.json`, `operating_point_valdo_letzter_stand_s1.json`, `operating_point_valdo_ens2_letzter_stand.json`, `ens2_letzter_stand_valdo.json`):**
| feste Zelle 0.3 / 2 mm3 | Zwischenstandswahl s42 | s1 | Abstand | Mittel | letzter Stand s42 | s1 | Abstand | Mittel |
|---|---|---|---|---|---|---|---|---|
| roh | 0.585 | 0.553 | 0.032 | 0.569 | 0.538 | 0.545 | 0.007 | 0.542 |
| mit Liquor-Regel | 0.646 | 0.590 | 0.056 | 0.618 | 0.594 | 0.614 | 0.020 | 0.604 |
Gepaart je Startwert: s42 -0.047 [-0.088; -0.010] (roh), s1 -0.007 [-0.068; +0.051] (roh); mit Liquor-Regel und kreuzweise gewaehlter Schwelle s42 -0.043 [-0.096; +0.011], s1 **+0.065 [+0.012; +0.126]** -- die Vorzeichen sind ENTGEGENGESETZT.
In beiden Startwerten ist der letzte Stand empfindlicher und unpraeziser (s1: S 0.71 gegen 0.60, P 0.44 gegen 0.51, 2.19 gegen 1.42 Fehlalarme je Fall), sein bestes Feld liegt bei Schwelle 0.5-0.6 statt 0.3.
Zwei-Startwerte-Ensemble (das VALDO-Standardrezept): letzter Stand 0.571 roh / 0.626 mit Regel gegen 0.594 / 0.647; gepaart roh -0.023 [-0.069; +0.012] (nachtraeglich gerechnet, nicht vorab festgelegt).
**Entscheidung nach der vorab festgelegten Regel:** der ABSTAND zwischen den Startwerten schrumpft wie vorhergesagt (0.032 -> 0.007 roh, 0.056 -> 0.020 mit Regel) -- die Zwischenstandswahl auf ~11 Faellen IST eine Rauschquelle zwischen Startwerten;
aber das MITTEL sinkt (0.569 -> 0.542 roh, 0.618 -> 0.604 mit Regel), und das Ensemble wird nicht besser. Bedingung 2 verletzt -> **die Zwischenstandswahl bleibt im Rezept.**
**Was daraus fuer die Berichterstattung folgt (wichtiger als die Rezeptfrage):** (1) die 0.646 von Startwert 42 sind zum Teil ein Losgewinn bei der Zwischenstandswahl -- mit dem letzten Stand faellt derselbe Lauf auf 0.594, Startwert 1 steigt von 0.590
auf 0.614. Beide Verfahren liefern im Mittel ~0.60-0.62; das deckt sich mit dem Mittel 0.609 ueber sechs nahezu gleiche Laeufe (Abschnitt 3). Im Paper gehoert deshalb das MITTEL ueber Startwerte mit Spanne in die Haupttabelle, nicht der beste Lauf.
(2) Die feste Zelle (Schwelle 0.3) ist an Modellen mit Zwischenstandswahl gewaehlt und benachteiligt spaete, selbstsichere Zwischenstaende; wer das Rezept aendert, muss den Arbeitspunkt mitpruefen. (3) Fuer die Frage "120 Epochen generell?" (5h):
laengeres Training hilft hier nur ZUSAMMEN mit der Zwischenstandswahl. **Grenzen:** zwei Startwerte; die Mittelwert-Differenz (-0.014 mit Regel) liegt unter dem Rauschboden von 0.03 -- "schlechter" ist fuer das Mittel nicht gesichert, "nicht besser" schon.

### 5p Augmentierung, sauber getrennt (Nutzerfrage 21.09. abends; eingereiht)
**Stand:** das Rezept spiegelt nur (drei Achsen) und legt die Blutung zufaellig in den Ausschnitt. Der einzige bisherige Test (4.7) war ein PAKET aus z-Normierung und Augmentierung (Drehung, Massstab, Kontrast, Rauschen) auf dem alten, uebermalten
Gitter: -0.060, nicht trennbar. Augmentierung ALLEIN ist also nie gemessen worden. **Was es gibt und was zu unserem Problem passt:** (a) raeumlich (Drehung, Massstab, elastisch) -- interpoliert die wenigen Voxel einer Blutung; unser eigener
Befund 4.8 (Umrechnung kostet 15-30 % Kontrast) und die Warnung im MicrobleedNet-Paper sprechen dagegen; (b) NUR Intensitaet (Gamma, glattes Feld, Kontrast, Rauschen) -- nichts wird interpoliert; zielt auf Skalen- und Scannerunterschiede, also
eher auf die Uebertragung zwischen Kohorten (5n) als auf die CV innerhalb einer Kohorte; (c) DICKSCHICHT-SIMULATION -- duennschichtige Faelle werden in z gemittelt und zurueckgerechnet, das Label folgt; zielt genau auf unsere schwaechsten
Daten (VALDO-3 F1 0.479 und VALDO-1 0.557 gegen 0.634 bei duennen Schichten; catalina-T2* mit 5 mm); (d) kuenstliche Blutungen und Verwechsler (CenSynCMB +2.6 Punkte) -- aufwendig, zurueckgestellt; (e) nnU-Net-Standardsatz (alles zusammen,
dort mit 1000 Epochen) -- Augmentierung und lange Trainingsdauer gehoeren zusammen (5h). **Aenderung (`code/turnier.py`, Sicherung .vor-aug2-0921):** `--aug-intensitaet` (-augi) und `--aug-schicht` (-augs), jeweils EINZELN schaltbar.
**Pruefung (CPU, kuenstliche Ein-Schicht-Blutung):** Intensitaet -- Label und FRST-Kanal bitgleich, Hintergrund bleibt 0, 38 von 40 Ausschnitten veraendert; Dickschicht -- Blutung von 1 Schicht wird 2-5 Schichten (= simuliertes k), das Profil
durch die Blutung 0.95 -> 0.62 / 0.71 / 0.70 / 0.60 (Partialvolumen wie bei echten dicken Schichten), Hintergrund bleibt 0. **Experiment:** Basisrezept + `--aug-schicht` auf VALDO (eingereiht); Auswertung gepaart an der festen Zelle UND je
Kohorte (`evaluate.py --split-valdo`) -- die Hypothese sagt einen Gewinn in den Kohorten 1 und 3 voraus, nicht in Kohorte 2.
**Zweites Experiment (Nutzerfreigabe 21.09. 20:38, eingereiht als [160]):** Basisrezept + `--aug-intensitaet` auf VALDO -> `ergebnisse/turnier/a03-aniso-e60-gd-augi`. **Vorab festgelegt, bevor ein Ergebnis vorliegt:**
(1) INNERHALB von VALDO, gepaart an der festen Zelle gegen die Basis (Startwert 42): erwartet wird Gleichstand, KEIN Gewinn -- die CV-Faelle stammen aus denselben drei Scannern wie das Training. Die Aenderung gilt als unschaedlich, wenn das
gepaarte Intervall die Null enthaelt; zur Einordnung: zwei Startwerte der Basis liegen 0.032 auseinander (0.585 gegen 0.553), ein Unterschied in dieser Groesse ist ohne zweiten Startwert nicht deutbar.
(2) UEBERTRAGUNG ohne Anpassung (Zero-shot, `cmb.transfer.zero_shot`, Mittel der fuenf Faltenmodelle) auf catalina-T2* und catalina-SWI, gepaart ueber die Faelle gegen den vorhandenen Zero-shot der Basis (T2* 0.622, SWI 0.305). HIER sagt die
Hypothese einen Gewinn voraus, und zwar ueber WENIGER FEHLALARME je Fall (5n: die Uebertragung kostet ueber Fehlalarme, nicht ueber den Markierungsstil; SWI scheitert an Kontrast und Skala). Bestaetigt, wenn das gepaarte Intervall in
mindestens einer Kohorte ueber Null liegt und in der anderen nicht gesichert darunter; sonst widerlegt. Dieselben zwei Zero-shot-Laeufe bekommt `-augs` (Dickschicht; catalina-T2* hat 5-mm-Schichten) -- eingereiht [161]-[164].
(3) RAUSCHMASS: der Zero-shot-Wert haengt auch am Startwert des Trainings; dafuer wird die Basis mit Startwert 1 ebenfalls ohne Anpassung uebertragen ([165], [166]). Ein Augmentierungsgewinn zaehlt nur, wenn er groesser ist als der Abstand
zwischen den beiden Basis-Startwerten. **Folge:** bestaetigt -> die Augmentierung kommt in das Rezept fuer Uebertragungslaeufe, naechster Test ist das Feintuning vom `-augi`-Start; widerlegt -> gestrichen, negativer Befund bleibt stehen.
**Ergebnis Dickschicht-Simulation innerhalb VALDO (22.09. 13:34; gewaehlte Zwischenstaende 55 / 50 / 55 / 55 / 45; `ergebnisse/analysis/valdo_augs_*.json`, `operating_point_valdo_augs.json`): die Hypothese ist WIDERLEGT.**
Feste Zelle roh **0.458 gegen 0.585, gepaart -0.127 [-0.179; -0.075]** (ueber dem Rauschmass 0.08 von Abschnitt 3); P 0.34 gegen 0.52, S 0.70 gegen 0.67, Fehlalarme je Fall 3.3 gegen 1.5. Mit Liquor-Regel 0.522 gegen 0.646; mit eigener,
kreuzweise gewaehlter Schwelle (0.6-0.7 statt 0.3) 0.573 gegen 0.622, -0.050 [-0.115; +0.024]. Je Kohorte: 4 mm -0.045 (n = 8, n. s.), 0.8 mm -0.125 [-0.198; -0.063], **3-4 mm -0.146 [-0.284; -0.012]** -- die Hypothese sagte fuer die
dicken Kohorten einen GEWINN voraus; eingetreten ist dort der groesste Verlust (VALDO-3: 3.5 Fehlalarme je Fall, P 0.21). **Interpretiert:** das Netz, das in 35 % der Ausschnitte kuenstlich verschmierte Blutungen mit gestrecktem Label lernt,
wird selbstsicherer und unpraeziser -- es lernt "auch Schwaches und Ausgedehntes ist Blutung", und genau das sind die Verwechsler in den dicken Kohorten (Gefaessquerschnitte, Partialvolumen von Sulci). ~~Die Simulation erzeugt Bilder, die den dicken Kohorten aehneln, aber Labels, die es dort nicht gibt (der Referenz-Befunder markiert auf der dicken Schicht klein, nicht ausgedehnt).~~
**BERICHTIGT am 22.09. 21:24 -- diese Erklaerung war falsch und ist nachgemessen widerlegt.** Ausdehnung der Referenz-Blutungen in z auf dem gemeinsamen Gitter (Median [10.-90. Perzentil], Voxel a 1 mm): Kohorte 1 (4 mm) **4.0 [4; 8]**,
Kohorte 2 (0.8 mm) 3.0 [2; 4], Kohorte 3 (3-4 mm) **4.0 [3; 7]** -- reale Dickschicht-Annotationen SIND auf dem Gitter in z ausgedehnt, und zwar genau in der Groessenordnung, die die Simulation erzeugt (k = 2..5 streckt 3 Voxel auf 4-8).
Die Etikett-Behandlung der Augmentierung war also richtig; meine Erklaerung war eine Plausibilitaets-Erzaehlung, die ich nicht gemessen hatte -- derselbe Fehler wie der, den sie erklaeren sollte.
**Die tatsaechliche Erklaerung liefert die Robustheitsprobe (5v):** das Netz ist gegen z-Verschmierung extrem empfindlich (Sensitivitaet 0.66 -> 0.43 -> 0.25 -> 0.17 -> 0.13 fuer k = 2..5, bei STEIGENDER Precision 0.63 -> 0.86). In 35 % der
Trainingsausschnitte verschmierte Blutungen zwingen das Netz also, schwaechere und diffusere Belege zu akzeptieren, um sie ueberhaupt noch zu finden. Genau diese gesenkte Latte feuert dann auf scharfen Testbildern auf jede diffuse Struktur:
S 0.70 (hoeher als die Basis), P 0.34, 3.3 Fehlalarme je Fall. Dazu passt, dass der Lauf an SEINEM eigenen Arbeitspunkt (Schwelle 0.6-0.7 statt 0.3) nur noch -0.050 [-0.115; +0.024] hinten liegt: ein grosser Teil des Rueckstands ist
Kalibrierung, nicht Koennen. **Die Lehre bleibt, aber anders formuliert:** eine Augmentierung, die die Aufgabe im Training schwerer macht, verschiebt den Arbeitspunkt des Netzes -- wer sie einsetzt, muss den Arbeitspunkt mitbestimmen
(oder je Modell neu waehlen) und auf scharfen Daten mit mehr Fehlalarmen rechnen. **Entscheidung:** innerhalb VALDO verworfen. Die Zero-shot-Laeufe auf catalina bleiben eingereiht (dort ist das Hauptmass vorab festgelegt) -- Erwartung nach diesem Befund: auch dort mehr Fehlalarme, kein Gewinn.
**Ergebnis Intensitaets-Augmentierung innerhalb VALDO (22.09. 15:18; gewaehlte Zwischenstaende 35 / 60 / 20 / 55 / 35; `ergebnisse/analysis/valdo_augi_*.json`, `operating_point_valdo_augi.json`): erwartet war Gleichstand, gemessen ist
ein Verlust ueber dem Rauschmass.** Feste Zelle roh **0.477 gegen 0.585, gepaart -0.108 [-0.180; -0.030]**; mit Liquor-Regel 0.527 gegen 0.646; kreuzweise 0.458 gegen 0.622. Je Kohorte -0.139 [-0.246; -0.065] / -0.113 / -0.058. Anders als bei der
Dickschicht-Simulation fallen hier BEIDE Achsen: S 0.56 gegen 0.67 UND P 0.41 gegen 0.52 (1.9 Fehlalarme je Fall) -- das Netz hat bei gleichem Budget weniger gelernt, nicht mehr Falsches. Der Lauf liegt unter dem schwaechsten der drei
Startwerte (0.501). **Interpretiert:** fuenf Operationen mit je p = 0.5 treffen einen Ausschnitt im Mittel 2.5-fach; Gamma auf der invertierten Skala veraendert den Laesionskontrast systematisch (g < 1 hebt den Hintergrund, g > 1 drueckt ihn);
zusammen ist die Stoerung fuer 60 Epochen und 35 Trainingsfaelle offenbar zu stark -- das Netz normiert per Instanz ohnehin und gewinnt aus der Streuung nichts, zahlt aber mit Lernbudget. Ob die Basis gegen solche Stoerungen bereits invariant ist
(dann kann Training damit nur kosten), misst die Robustheitsprobe (5v) heute Nacht. **Stand der Augmentierungsfrage:** beide Varianten sind innerhalb VALDO gesichert schlechter (Dickschicht -0.127, Intensitaet -0.108); das Hauptmass fuer
die Uebertragung (Zero-shot auf catalina, 5p (2)) laeuft als Naechstes -- nach S 0.56 innerhalb der Domaene erwarte ich dort keinen Gewinn.
**Hauptmass: Uebertragung ohne Anpassung auf catalina (22.09. 16:37; `mb-arena/transfer/auswertung_zeroshot_t2star_aug_0922.json`, `..._swi_aug_0922.json`; Summen):**
| Zero-shot (Mittel der 5 Faltenmodelle) | catalina-T2* F1 (FP je Fall) | gepaart gegen Basis s42 | catalina-SWI F1 (FP je Fall) | gepaart |
|---|---|---|---|---|
| Basis s42 | 0.622 (4.8) | -- | 0.305 (6.7) | -- |
| Basis s1 = Rauschmass | 0.638 (4.0) | +0.016 [+0.003; +0.033] | 0.311 (5.4) | +0.006 [-0.015; +0.030] |
| Intensitaets-Augmentierung | 0.608 (5.9) | -0.014 [-0.034; +0.000] | 0.297 (9.0) | -0.007 [-0.029; +0.013] |
| Dickschicht-Simulation | 0.606 (5.5) | -0.016 [-0.037; +0.002] | 0.240 (16.3) | **-0.065 [-0.096; -0.036]** |
**Entscheidung nach der Regel aus (2) und (3): beide Hypothesen WIDERLEGT.** Kein Intervall liegt ueber Null; der vorhergesagte Mechanismus (weniger Fehlalarme je Fall) tritt nirgends ein -- beide Augmentierungen ERHOEHEN die Fehlalarme in
beiden Kohorten (T2* 4.8 -> 5.5-5.9; SWI 6.7 -> 9.0 bzw. 16.3), die Dickschicht-Simulation auf SWI gesichert und massiv. Das Rauschmass der Uebertragung ist mit +0.016 / +0.006 klein; beide Augmentierungen liegen mit umgekehrtem Vorzeichen
in derselben Groesse. **Nebenbefund, der fuer die Auslieferung zaehlt:** Startwert 1 ist innerhalb VALDO der schwaechere (0.553 gegen 0.585), uebertraegt aber BESSER (0.638 gegen 0.622, gesichert; 4.0 statt 4.8 Fehlalarme) -- die CV-Rangfolge
der Startwerte sagt die Uebertragung nicht voraus. Das stuetzt das Zwei-Startwerte-Ensemble als ausgeliefertes Modell und warnt davor, "den besten Startwert" zu waehlen. **Bilanz 5p:** drei vorab festgelegte Messungen, drei negative
Befunde (VALDO -0.127 / -0.108, Uebertragung 0 bis -0.065). Augmentierung in dieser Form ist gestrichen; ob ueberhaupt eine Stoerung eine echte Schwaeche des Netzes trifft, beantwortet die Robustheitsprobe (5v) VOR jedem weiteren Versuch.
(Nummern zum Zeitpunkt des Einreihens; seit dem Vorziehen der zweiten Stufe am 21.09. 21:52 jeweils um 2 hoeher.) **Grenzen:** ein Startwert je Variante; die Staerken (Gamma 0.7-1.5, Feld exp(N(0, 0.15)), Kontrast 0.85-1.15) sind gesetzt, nicht abgestimmt -- ein Nullbefund widerlegt diese Einstellung, nicht die Idee.

### 5q Zwischenbilanz: was gelernt ist und welcher Suchraum bleibt (Nutzerfrage 21.09. 21:00)
**Gelernt (jeweils mit Beleg):**
(1) DATEN SCHLAGEN METHODE. Alle grossen Gewinne lagen in der Datenkette: Original-Uebermalung weg +0.136 gesichert (4.1), FRST behalten +0.053 (4.11), z = 1 mm +0.036 (4.8), Liquor-Regel +0.02 bis +0.06 (4.3), fester Arbeitspunkt (5j).
(2) KEIN TRAININGSHEBEL HEBT DAS F1: blob -0.044, Zentrumskarte -0.054, fnrw -0.030, harte Negative +-0, Verhaeltnis 1:2 kollabiert, Gewebekanaele -0.058, eigene Uebermalung ohne Gewinn (5g, 5j-5m). Das Muster ist immer dasselbe: empfindlicher
(kleine Blutungen 0.52 -> 0.64 / 0.66), bezahlt mit Fehlalarmen. Der Engpass ist nicht das Finden, sondern das TRENNEN von Blutung und Verwechsler mit nur einer Sequenz. Dazu passt: bei CenSynCMB halfen genau diese zwei Hebel (FN-Neugewichtung
+1.8, Zentrumskarte +4.5 Punkte) -- dort mit T1 + T2 + T2*. Empfindlichkeit lohnt nur, wenn etwas da ist, das die zusaetzlichen Kandidaten trennen kann.
(3) RAUSCHBODEN ~0.03: sechs nahezu gleiche Laeufe 0.590-0.646 (Abschnitt 3); ein Gewinn des Wald-Klassifikators loeste sich mit unabhaengiger erster Stufe auf (+0.033 -> +0.004), der des 3D-CNN bestaetigte sich (+0.044 / +0.045). Ohne
zweiten Startwert bzw. unabhaengige erste Stufe ist nichts unter 0.03 deutbar.
(4) MISCHEN HILFT AUF KANDIDATEN-EBENE, NICHT AUF VOXEL-EBENE: U-Nets je Kohorte; gemischt oder vortrainiert + feingetunt schlechter (-0.02 bis -0.045, ueber Fehlalarme, 5h, 5n); EIN gemeinsamer Klassifikator ueber alle Kohorten: SWI +0.045,
reproduziert (5j; Idee des Nutzers).
(5) WENIGE FAELLE TRAGEN DIE GEPOOLTEN ZAHLEN: drei catalina-T2*-Faelle = 28 % der Blutungen und rund die Haelfte der Uebersehenen (5i); ohne sie F1 0.696.
(6) Meine Erwartungen lagen mehrfach falsch (Mischen, Zentrumskarte, Feintuning vom gemischten Netz, "Kleinhirn-Luecke") -> vorab festlegen, gepaart messen, Faelle zaehlen.

**Suchraum -- ausgeschoepft (nicht wieder anfassen ohne neuen Grund):** Verlustfunktionen und instanzbewusstes Training, Stichprobenverhaeltnis, harte Negative, Gewebe als Netz-Eingabe, Uebermalung in jeder Dosis, Mischstrategien, Vortraining +
Feintuning, zweite Stufe JE KOHORTE (logistisch, Wald, Boosting, CNN), Ensembles ueber zwei Mitglieder hinaus, Spiegel-TTA auf catalina.
**Laeuft (Schlange, danach ist das Trainingsrezept vermessen):** letzter Zwischenstand (5o), 120 Epochen (5h), Ausschnittgroesse, Augmentierung einzeln (5p), dritter Startwert.
**Offen, geordnet nach erwartetem Ertrag je Aufwand:**
(a) ETIKETT-DECKE MESSEN (keine GPU; Expertenurteil des Nutzers auf der Pruefseite 5b, 201 Kandidaten): sind ein Teil der "Fehlalarme" echte Blutungen, liegt die Precision hoeher als gemessen, und es wird sichtbar, wie nah 0.65 an dem
liegt, was die Referenz hergibt. Das VALDO-Paper berichtet keine Uebereinstimmung zwischen Befundern (Abschnitt 6). Ohne diese Zahl optimieren wir moeglicherweise gegen Etikettrauschen.
(b) ZWEITE STUFE AUSBAUEN (einzige Richtung mit reproduziertem Gewinn; billig, 216 081 Parameter): groesserer oder laenglicher Kontext bzw. MinIP-Ansichten (Verwechsler sind vor allem Gefaesse und an ihrer Fortsetzung ueber Schichten
erkennbar); Kandidaten mehrerer Startwerte der ersten Stufe als Training; Uneinigkeit der Startwerte bzw. TTA-Streuung als Merkmal; vortrainierter 3D-Encoder oder echtes ResNet; Bestaetigung auf VALDO / T2* mit zweitem Startwert (steht aus).
(c) KUENSTLICHE BLUTUNGEN UND VERWECHSLER (CenSynCMB: +1.4 und +1.2 Punkte auf genau diesem Datensatz; der einzige Literaturhebel, den wir nicht gemessen haben). Fuer die ZWEITE Stufe viel billiger als fuer die erste: es braucht nur
Ausschnitte, keine ganzen Volumen, und es trifft genau den Engpass (mehr harte Negative und Positive fuer den Klassifikator). Aufwand mehrere Tage, kein Code veroeffentlicht.
(d) ARCHITEKTUR AUF SAUBEREN DATEN NACHMESSEN: das Turnier (4.4) lief mit 40 Epochen und Fruehstopp auf dem ALTEN, uebermalten Gitter -- die Rangfolge (aniso gewinnt, grosse Netze "nicht trainierbar") stammt aus beschaedigten Daten und
kurzem Budget. Erwarteter Gewinn klein, aber eine Gutachterfrage; 2-3 Netze x 120 Ep. ~ 10-15 GPU-Stunden.
(e) OBERGRENZE DER EIN-SEQUENZ-LOESUNG (nur VALDO): T1 + T2 + T2* als Eingabe. Fuer catalina nicht nutzbar (kein T1, FLAIR nur Untergruppe, 4.10), liefert aber die Zahl, wie viel des Abstands zu CenSynCMB 74.3 an den Sequenzen liegt.
(f) KLEINERE POSTEN: Perzentil-Skala p99.5 (+0.025 n. s., einziges positives Vorzeichen der Vorverarbeitung, zweiter Startwert am 21.09. gestrichen, 4.9); Training in nativer Aufloesung statt Umrechnung (4.8: Umrechnung kostet 15-30 %
Kontrast; aufwendig, Ausgang offen); Schwelle je Kohorte. NICHT verfuegbar: Phasenbild zur Trennung Kalk / Blut (VALDO liefert keins; fuer catalina-SWI beim Nutzer zu erfragen).
**Empfehlung:** (a) zuerst, weil es jede weitere Zahl einordnet; dann (b) und (c) zusammen -- kuenstliche Verwechsler nur fuer den Klassifikator; (d) in GPU-Luecken. Nichts davon ist begonnen; diese Liste ist ein Plan, kein Ergebnis.

### 5r Zweite Stufe ausbauen: mehr Kontext (5q b; Nutzerfreigabe 21.09. abends; eingereiht)
**Original:** MicrobleedNet prueft Kandidaten mit einem zweiten Netz auf 24^3-Ausschnitten. **Stand bei uns:** das gemeinsame kleine 3D-CNN (5j) sieht um jeden Kandidaten einen Wuerfel von 16 mm (16 x 32 x 32 Voxel bei 1 x 0.5 x 0.5 mm);
gespeichert waren 20 mm. **Beobachtung:** die Verwechsler sind vor allem Gefaesse, und ein Gefaess verraet sich durch seine FORTSETZUNG -- durch die Schichten oder in der Ebene. Bei 4-5 mm Schichtdicke (VALDO-1 / -3, catalina-T2*) enthaelt
der 16-mm-Wuerfel nur 3-4 gemessene Schichten. **Hypothese:** ein zweiter, grober Blick (28 mm bei halber Aufloesung) senkt die Fehlalarme bei gleicher Sensitivitaet, am staerksten in den Kohorten mit dicken Schichten.
**Aenderung:** `candidates.py --half 32 32 16` speichert 32-mm-Wuerfel (Vorgabe unveraendert 20 mm); `classifier.fit_predict_cnn_two_scale` = derselbe Encoder ZWEIMAL (fein: mittlere 16 mm in voller Aufloesung; grob: 28 mm, vorab 2 x
gemittelt), die zwei 64er-Merkmalsvektoren werden vor der linearen Schicht verkettet; Verlust, Optimierer, 40 Epochen, drei Mitglieder, Spiegelung und Versatz wie beim feinen CNN, damit der Vergleich nur den Kontext misst;
`pooled.py --models cnn cnn2s`. Sicherungen `*.vor-zweiskalig-0921`. **Pruefung (CPU):** (1) alte Ausschnittgroesse: alte und neue Fassung des feinen CNN liefern bitgleiche Werte; (2) auf den grossen Ausschnitten liefert das feine CNN exakt
dasselbe wie auf deren mittleren 20 x 40 x 40 -- die bisherigen Zahlen bleiben also reproduzierbar; (3) aendert man nur den Rand AUSSERHALB des feinen Blicks, reagiert das zweiskalige Netz, das feine nicht; (4) zu kleine Ausschnitte
werden abgelehnt. **Experiment:** EIN Lauf rechnet `pooled cnn` und `pooled cnn2s` auf denselben Kandidaten (VALDO, T2*, SWI; Kandidaten wie bisher, nur mit groesserem Ausschnitt). **Vorab festgelegt:** Hauptmass ist die gepaarte Differenz
cnn2s - cnn je Kohorte (Bootstrap ueber Faelle; `cmb.stage2.compare_runs`). Uebernommen wird cnn2s nur, wenn die Differenz in mindestens einer Kohorte gesichert ueber Null liegt und in keiner gesichert darunter, UND wenn sich das mit der
unabhaengigen SWI-Stufe (Startwert 1) wiederholt -- dieselbe Huerde, die das feine CNN genommen hat und der Wald-Klassifikator nicht (5j). Eingebaute Kontrolle: `pooled cnn` muss die bisherigen Werte treffen (0.625-0.636 / 0.672 / 0.670;
die GPU rechnet nicht bitgleich, zwei fruehere Laeufe lagen 0.005 auseinander). **Grenze:** ein Startwert der zweiten Stufe je Lauf (drei gemittelte Mitglieder); die Gruende der Fehlalarme kennen wir erst mit dem Expertenurteil (5b) --
sagt es "meist Gefaesse", stuetzt das den Ansatz, sagt es "meist echte Blutungen", ist die zweite Stufe das falsche Werkzeug.
**Ergebnis Hauptlauf (22.09. 00:08; GPU 67 min; `mb-arena/stage2/pooled_h32_cnn_cnn2s_0921.json`):** eingebaute Kontrolle bestanden -- `pooled cnn` trifft auf den grossen Ausschnitten die alten Werte (0.626 / 0.672 / 0.670 gegen 0.625-0.630 / 0.672 / 0.670).
| | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| nur Schwelle | 0.565 | 0.654 | 0.626 |
| feines CNN (16 mm) | 0.626 (P 0.63, S 0.63, 0.91 FP je Fall) | 0.672 (P 0.70, S 0.65, 3.04) | 0.670 (P 0.74, S 0.61, 2.26) |
| zweiskalig (16 + 28 mm) | 0.629 (P 0.66, S 0.60, 0.77) | **0.687** (P 0.73, S 0.65, 2.71) | **0.686** (P 0.74, S 0.64, 2.46) |
| **gepaart zweiskalig - fein (Hauptmass)** | +0.003 [-0.026; +0.036] | +0.016 [-0.002; +0.030] | +0.016 [-0.002; +0.037] |
| gepaart zweiskalig - nur Schwelle | +0.064 [+0.017; +0.105] | +0.033 [+0.017; +0.055] | +0.060 [+0.035; +0.094] |
Je VALDO-Kohorte (zweiskalig - fein): 4-mm-Schichten (n = 8) -0.028 [-0.110; +0.000]; 0.8 mm (n = 27) -0.008 [-0.054; +0.039]; 3-4 mm (n = 22) **+0.057 [+0.000; +0.107]** (Fehlalarme 19 -> 13 bei gleichen 17 Treffern).
**Gelesen gegen die vorab festgelegte Regel:** das Hauptmass ist in KEINER Kohorte gesichert (zweimal +0.016 mit unterer Grenze -0.002) -> Bedingung 1 ist knapp NICHT erfuellt; uebernommen wird vorerst nichts. Die Richtung passt zur Hypothese
dort, wo sie es vorhersagt: auf T2* (5-mm-Schichten) und VALDO-3 sinken die Fehlalarme bei gleicher Sensitivitaet; auf SWI kommt der Zuwachs dagegen ueber die Sensitivitaet (anderer Mechanismus als vorhergesagt), auf VALDO-1 (n = 8) zeigt er in
die falsche Richtung. Gegen die reine Schwelle ist die zweiskalige Stufe erstmals in ALLEN drei Kohorten gesichert besser (das feine CNN war es auf VALDO nicht). Der Bestaetigungslauf mit der unabhaengigen SWI-Stufe laeuft; erst er entscheidet.
**Bestaetigungslauf und Entscheidung (22.09. 01:16; `mb-arena/stage2/pooled_h32_cnn_cnn2s_swi_s1_0921.json`; SWI-Kandidaten der unabhaengig trainierten ersten Stufe, VALDO- und T2*-Kandidaten unveraendert):**
| | VALDO | catalina-T2* | catalina-SWI (erste Stufe Startwert 1) |
|---|---|---|---|
| feines CNN | 0.632 (P 0.66, S 0.60, 0.75 FP je Fall) | 0.672 (P 0.71, S 0.64, 2.87) | 0.665 (P 0.73, S 0.61, 2.48) |
| zweiskalig | 0.601 (P 0.61, S 0.60, 0.95) | 0.691 (P 0.74, S 0.64, 2.43) | 0.678 (P 0.75, S 0.62, 2.26) |
| **gepaart zweiskalig - fein** | **-0.030 [-0.059; -0.011]** | **+0.019 [+0.001; +0.036]** | +0.013 [-0.001; +0.031] |
**Entscheidung nach der vorab festgelegten Regel: NICHT uebernommen.** Im Hauptlauf war keine Kohorte gesichert, im Bestaetigungslauf ist T2* gesichert besser, VALDO aber gesichert SCHLECHTER -- die Bedingung "in keiner Kohorte gesichert darunter"
ist verletzt. Die Endkette behaelt das feine CNN. **Was bleibt:** (1) auf catalina-T2* (5-mm-Schichten) ist das Bild in beiden Laeufen dasselbe -- +0.016 / +0.019, Fehlalarme 3.04 -> 2.71 bzw. 2.87 -> 2.43 bei gleicher Sensitivitaet --, genau der
vorhergesagte Mechanismus in der Kohorte mit den dicksten Schichten; auf SWI zweimal positiv, zweimal nicht gesichert (+0.016 / +0.013). (2) **Die zweite Stufe hat ein eigenes Trainingsrauschen, das unsere Intervalle NICHT enthalten:** zwischen den
beiden Laeufen sind die VALDO-Kandidaten identisch, geaendert hat sich nur der SWI-Teil des Trainingspools -- trotzdem faellt das zweiskalige Netz auf VALDO von 0.629 auf 0.601, das feine geht von 0.626 auf 0.632. Der Bootstrap ueber Faelle misst
die Unsicherheit der FALLAUSWAHL bei EINEM trainierten Klassifikator, nicht die des Trainings; ein "gesichertes" -0.030 ist deshalb weniger sicher, als das Intervall aussieht (dasselbe gilt rueckwirkend fuer die +0.044 / +0.045 des feinen CNN auf
SWI -- dort allerdings zweimal mit unabhaengiger erster Stufe gleich ausgefallen). Sauber waere: jede Variante der zweiten Stufe 5 x mit verschiedenen Startwerten trainieren und die Streuung mitberichten (~6 GPU-Stunden je Vergleich).
(3) Ein nachtraeglicher Gedanke, KEINE Entscheidung: den groben Blick nur fuer Dickschicht-Kohorten zu nutzen widerspraeche der Idee EINES Klassifikators und waere nach Ansicht der Daten gewaehlt. **Empfehlung:** kein weiterer GPU-Aufwand fuer
den Kontext; der erwartbare Gewinn (<= 0.02) liegt unter dem Rauschen. Aussichtsreicher bleiben die Etikett-Decke (5b) und kuenstliche Verwechsler (5q c).
**Noch nicht begonnen (5q b, c):** Kandidaten mehrerer Startwerte als Trainingsmaterial (fuer T2* fehlt der zweite Startwert), Uneinigkeit der Startwerte als Merkmal, vortrainierter Encoder, kuenstliche Blutungen und Verwechsler fuer den
Klassifikator (Entwurf folgt, sobald 5b und dieser Lauf zeigen, WELCHE Verwechsler fehlen).

### 5t Ausschnittgroesse als Dosisreihe (Nutzerfrage 21.09. frueh zu Sundaresans Ausschnitt-System; kleiner Ausschnitt fertig 22.09. 07:43)
**Original:** MicrobleedNet trainiert die Kandidatenstufe auf 48^3-Voxel-Wuerfeln des Originalgitters (bei duennen Schichten ~24-48 mm). **Bei uns:** 38 x 62 x 94 Voxel auf dem 0.5 x 0.5 x 1 mm-Gitter = 38 x 31 x 47 mm, 800 je Epoche, Schiebefenster
in der Vorhersage. **Frage:** bringt Sundaresans kleiner Ausschnitt etwas (mehr Ausschnitte je Fall, weniger Hintergrund je Ausschnitt), oder kostet der fehlende Kontext? **Reihe:** klein 24 x 48 x 48 Voxel (24 x 24 x 24 mm) -- Rezept -- gross
64 x 128 x 128 (64 x 64 x 64 mm, eingereiht); sonst alles unveraendert, auch die 800 Ausschnitte je Epoche.
**Ergebnis klein (`ergebnisse/analysis/valdo_patch24_*.json`, `operating_point_valdo_patch24.json`; je Falte ~10 statt ~21 min):** feste Zelle roh **0.369 gegen 0.585, gepaart -0.216 [-0.277; -0.156]**; P 0.25 gegen 0.52, S 0.74 gegen 0.67,
Fehlalarme je Fall 5.5 gegen 1.5. Mit Liquor-Regel 0.491 gegen 0.646; auch mit eigener, kreuzweise gewaehlter Schwelle (0.4-0.6 statt 0.3) nur 0.523 gegen 0.622, gepaart -0.099 [-0.170; -0.040]. Je Kohorte -0.080 (4 mm, n. s.) / -0.186 (0.8 mm)
/ -0.267 (3-4 mm), beide gesichert; am schlimmsten in der Kohorte mit den groessten Voxeln und dem meisten Nicht-Hirn in der Maske (VALDO-3: 6.8 Fehlalarme je Fall).
**Interpretiert:** der kleine Ausschnitt findet mehr (S 0.74) und trennt viel schlechter -- das Netz sieht 24 mm Kontext und kann Gefaesse, Sulci und Hirnrand nicht mehr von Blutungen unterscheiden; die Verwechsler sind gerade die
Strukturen, die sich erst ueber einige Zentimeter verraten. Das ist dasselbe Muster wie alle Empfindlichkeits-Hebel (5j-5m), nur staerker. **Zwei Vermengungen, ehrlich benannt:** (1) mit 800 Ausschnitten je Epoche sieht das Netz je Epoche
nur ein Viertel der Voxel (55 k gegen 221 k je Ausschnitt) -- weniger Kontext UND weniger Trainingsdaten je Epoche; nach 5h ist der Budgeteffekt aber klein gegen -0.216. (2) Sundaresans Rezept gleicht den kleinen Ausschnitt durch die ZWEITE Stufe
(24^3-Klassifikator) aus; wir haben hier nur die erste Stufe gemessen. Die Aussage lautet also: fuer ein EINSTUFIGES Netz ist ein 24-mm-Ausschnitt gesichert zu klein. **Entscheidung:** klein verworfen; das Rezept bleibt, bis der grosse Ausschnitt
gemessen ist (Regel vorab: gross wird nur uebernommen, wenn gepaart gesichert besser als das Rezept UND nicht schlechter in einer Kohorte; erwartet wird hoechstens ein kleiner Gewinn, weil der Sieger des Turniers ohnehin die anisotrope
erste Stufe nutzt und die Kontextgrenze eher bei den Verwechslern als beim Netz liegt).
**Grosser Ausschnitt 64 mm (gestartet 22.09. 07:39): in diesem Geruest nicht messbar.** Der Puffer je Epoche (800 Ausschnitte x 3 Kanaele x 1.05 M Voxel x 4 B, zweimal im Speicher) hebt den Prozess auf 33-35 GiB und damit ueber die
absichtliche Speichergrenze der Dienst-cgroup (MemoryHigh 32 GiB): Drosselung, 0 % GPU. Nicht die Grenze wird geaendert, sondern die Dosis: Ersatzpunkt **48 x 96 x 96 Voxel = 48-mm-Wuerfel** (2x Voxel der Basis, ~24 GiB), eingereiht.
Die Reihe lautet damit 24 mm -- Rezept (38 x 31 x 47 mm) -- 48 mm; 64 mm braeuchte halb so viele Ausschnitte je Epoche (Codeaenderung) und ~8 GPU-Stunden.
**Ergebnis 48 mm (22.09. 10:17; gewaehlte Zwischenstaende 60 / 45 / 20 / 45 / 45; je Falte ~29 min; `ergebnisse/analysis/valdo_patch48_*.json`, `operating_point_valdo_patch48.json`):** feste Zelle roh 0.569 gegen 0.585, gepaart
-0.016 [-0.074; +0.034]; P 0.49 gegen 0.52, S 0.67 gegen 0.67, Fehlalarme je Fall 1.67 gegen 1.51. Mit Liquor-Regel 0.609 gegen 0.646, kreuzweise 0.610 gegen 0.622 (-0.012 [-0.084; +0.050]). Je Kohorte -0.018 / -0.013 / -0.028, alle n. s.
**Dosisreihe komplett:** 24 mm 0.369 -- Rezept 38 x 31 x 47 mm **0.585** -- 48 mm 0.569 (roh); mit Liquor-Regel 0.491 -- **0.646** -- 0.609. **Entscheidung nach der vorab festgelegten Regel: 48 mm nicht uebernommen** (nicht gesichert besser;
in keiner Kohorte gesichert schlechter -- der Unterschied liegt im Rauschboden). **Interpretiert:** nach unten ist die Ausschnittgroesse ein scharfer Hebel (-0.216), nach oben ein stumpfer (-0.016 n. s.): ab etwa 3-4 cm Kontext hat das Netz,
was es fuer die Trennung braucht; mehr Kontext bringt bei 800 Ausschnitten je Epoche nichts, kostet aber die doppelte Rechenzeit (29 gegen 21 min je Falte) und Speicher. Sundaresans 48^3-Voxel-Wuerfel sind bei duennen Schichten
physikalisch naeher an unserem 24-mm-Punkt als an unserem Rezept; sein Rezept traegt den kleinen Ausschnitt durch die zweite Stufe. **Grenzen:** ein Startwert je Punkt (Rauschboden 0.03; die 48-mm-Differenz liegt darunter); die 800 Ausschnitte je
Epoche sind fest, der grosse Ausschnitt sieht also je Epoche doppelt so viele Voxel -- ein Vorteil, der ihm nichts genuetzt hat; 64 mm bleibt ungemessen (Speicher, siehe oben).

### 5v Welche Augmentierung passt zu CMB -- ein Verfahren statt Plausibilitaet (Nutzerfrage 22.09. 13:30; Werkzeug gebaut, GPU-Lauf eingereiht)
**Anlass:** 5p -- die Dickschicht-Simulation war plausibel begruendet, kostete 1.7 GPU-Stunden und schadete gesichert (-0.127), am staerksten dort, wo der Gewinn vorhergesagt war. Der Fehler war methodisch: Bild-Statistik der Zielkohorte
nachgeahmt, Etikett-Statistik nicht; und ueberhaupt erst nach dem Training gemessen. **Verfahren (fuenf Schritte, jeder vor dem naechsten Training):**
(1) *Invarianzen aufschreiben.* Was darf sich ohne Aenderung des Labels aendern: Orientierung ohne Interpolation (Spiegelung, 90-Grad-Drehung), globale Helligkeit, Bias-Feld, Rauschen, Kontrast. Was nicht: Groesse in mm, Rundheit, Kleinkontrast,
das Verhaeltnis der Schichten zueinander. Alles aus der zweiten Liste (Drehung mit Interpolation, Massstab, Dickschicht) ist verdaechtig, bevor es gemessen ist.
(2) *Robustheitsprobe am fertigen Netz, ohne Training* (`cmb/analysis/robustness_probe.py`, neu): die gespeicherten Faltenmodelle sagen ihre eigenen Testfaelle voraus, nachdem die BILDER im Speicher gestoert wurden -- sieben Stoerungen in
22 Staerken (Gamma 0.7-1.5, Bias-Feld exp(N(0, 0.05-0.3)), Kontrast x0.7-1.3, Rauschen 2-10 % der Hirn-Std, Dickschicht k = 2-5, Drehung 5 / 15 Grad, Massstab 0.9 / 1.1; geometrische Stoerungen werden auf Bild und FRST angewandt, die
Wahrscheinlichkeitskarte zurueckgedreht, das Label bleibt unberuehrt). Gewertet am festen Arbeitspunkt mit Liquor-Regel, gepaart gegen die ungestoerte Vorhersage DESSELBEN Laufs, alles out-of-fold. Lesart, vorab festgelegt: faellt das F1
bei realistischer Staerke (die Staerke, die die Zielkohorte tatsaechlich zeigt) deutlich -> das Netz ist nicht invariant, Training mit dieser Stoerung ist ein Kandidat; faellt es nicht -> Training damit bringt innerhalb der Domaene nichts;
faellt es schon bei milden, unrealistischen Staerken -> die Stoerung zerstoert das Signal, nicht trainieren. Kosten: Vorhersage statt Training, ~2 GPU-Stunden fuer alle 22 Bedingungen auf 57 Faellen (CPU-Rauchtest an einem Fall bestanden,
11 Bedingungen, Kette ungestoert = Basiswert). Eingereiht als [172] ans Ende der Schlange.
(3) *Realismus gegen die Zielkohorte messen* (`cohort_differences.py`, vorhanden): eine Augmentierung passt, wenn sie die VALDO-Statistik in Richtung catalina schiebt -- z. B. catalina-Blutungen sind blasser (relativer Kontrast 0.20-0.21 gegen
0.34-0.38). Eine Intensitaets-Augmentierung, die den Laesionskontrast SENKT, waere die passende; die eingereihte (`--aug-intensitaet`) streut nur um den Ausgangswert. Das wird nach dem Zero-shot-Ergebnis (5p) an den Statistiken geprueft.
(4) *Etikett-Pruefung:* nach der Augmentierung muss das Label aussehen wie echte Labels der Zieldomaene (Groesse in Voxeln, Ausdehnung in z). Genau daran ist die Dickschicht-Simulation gescheitert (Label 2-5 Schichten; reale Dickschicht-Labels
sind 1 Schicht).
(5) *Erst dann trainieren*, nur Kandidaten, die (2)-(4) bestehen; Gewinnort vorab festlegen; zweiter Startwert fuer alles unter 0.08 (Abschnitt 3); Ergebnis nach Mechanismus lesen (P / S / Fehlalarme), nicht nur nach F1 -- bei CMB ist jede
Augmentierung, die die Sensitivitaet hebt und die Precision senkt, ein Fehlalarm-Generator.
**Erwartung fuer die Probe (vorab):** Gamma / Kontrast / Rauschen bei milden Staerken ohne Effekt (das Netz normiert per Instanz), bei starken ein Abfall ueber Fehlalarme; Dickschicht ab k = 3 deutlich (die Kohorten 1 / 3 sind schon schwaecher);
Drehung 15 Grad und Massstab 0.9 / 1.1 deutlich (Interpolation kostet Kleinkontrast, 4.8). Trifft das zu, bleibt fuer die Uebertragung nur die photometrische Augmentierung in Zielrichtung (3) als Kandidat.
**Ergebnis (22.09. 21:24, 57 Faelle x 23 Bedingungen, 2.5 GPU-h; `ergebnisse/analysis/robustness_probe_valdo_s42.json`; ungestoert = 0.646, also die berichtete Zahl -- eingebaute Kontrolle bestanden):**
| Stoerung | Staerke | F1 | P | S | FP/Fall | gepaart gegen ungestoert |
|---|---|---|---|---|---|---|
| Gamma | 0.85 / 1.15 | 0.630 / 0.642 | 0.64 / 0.61 | 0.63 / 0.68 | 0.9 / 1.1 | -0.015 / -0.004 (n. s.) |
| Gamma | 1.5 | 0.583 | 0.49 | 0.72 | 1.8 | **-0.063 [-0.117; -0.020]** |
| Bias-Feld | 0.05 / 0.15 / 0.30 | 0.652 / 0.612 / 0.601 | | | | +0.007 / **-0.034 [-0.058; -0.005]** / -0.045 |
| Kontrast | 0.7 / 0.85 / 1.15 / 1.3 | 0.608 / 0.626 / 0.643 / 0.628 | | | | **-0.037 [-0.068; -0.006]** / -0.020 / -0.003 / -0.017 |
| Rauschen | 0.02 / 0.05 / 0.10 | 0.641 / 0.645 / 0.633 | | | | -0.005 / -0.000 / -0.013 (alle n. s.) |
| **Dickschicht** | k = 2 / 3 / 4 / 5 | 0.536 / 0.363 / 0.287 / 0.225 | 0.71 / 0.65 / 0.86 / 0.86 | **0.43 / 0.25 / 0.17 / 0.13** | 0.4 / 0.3 / 0.1 / 0.1 | **-0.110 / -0.283 / -0.358 / -0.421** |
| Drehung | 5 / 15 Grad | 0.645 / 0.627 | 0.73 / 0.76 | 0.58 / 0.53 | 0.5 / 0.4 | -0.000 / -0.018 (n. s.) |
| Massstab | 0.9 / 1.1 | 0.648 / 0.627 | 0.75 / 0.78 | 0.57 / 0.53 | 0.5 / 0.4 | +0.002 / -0.019 (n. s.) |
**Gegen die Erwartung gelesen:** (1) Photometrie -- bestaetigt. Milde Aenderungen prallen ab (das Netz normiert je Instanz), erst Gamma 1.5 kostet, und zwar ueber Fehlalarme (P 0.49). Das Netz ist gegen genau die Stoerungen invariant, mit denen
`--aug-intensitaet` trainiert hat (Gamma 0.7-1.5, Feld 0.15, Kontrast 0.85-1.15) -- Training damit konnte also nur Lernbudget kosten, was 5p gemessen hat. **Die Probe haette diesen Lauf gespart.**
(2) Dickschicht -- bestaetigt und staerker als erwartet: die mit Abstand groesste Schwaeche, k = 5 kostet 0.42 F1 und zwei Drittel der Sensitivitaet. Precision STEIGT dabei (0.63 -> 0.86): Verschmieren loescht Blutungen aus, es erfindet keine.
(3) Drehung und Massstab -- **meine Erwartung war falsch.** Das F1 bewegt sich nicht (alle vier n. s.), aber die Zusammensetzung kippt stark: Precision 0.63 -> 0.73-0.78, Sensitivitaet 0.66 -> 0.53-0.58, Fehlalarme 0.95 -> 0.37-0.51.
Die Interpolation loescht KLEINE STRUKTUREN UNTERSCHIEDSLOS -- Blutungen und Verwechsler gleichermassen --, und die beiden Verluste heben sich im F1 auf. Das ist die Messung zu 4.8 (Umrechnung kostet 15-30 % Kontrast): sie kostet rund
ein Achtel der Blutungen. **Methodisch:** "F1 unveraendert" heisst NICHT "invariant"; ohne P / S / FP haette ich hier genau falsch geschlossen. Die Regel "nach Mechanismus lesen, nicht nach F1" (5v Schritt 5) hat sich damit zum zweiten Mal
an diesem Tag bewaehrt.
**Was daraus folgt:** (a) Photometrische Augmentierung ist fuer dieses Netz erledigt -- es ist bereits invariant. (b) Die einzige echte Schwaeche ist die z-Aufloesung; sie erklaert die Kohorten-Rangfolge (0.634 duenn gegen 0.557 / 0.479 dick)
physikalisch statt methodisch. Ein naiver Trainingsversuch dagegen ist gescheitert (5p), weil er den Arbeitspunkt verschiebt; aussichtsreicher waeren ein Arbeitspunkt je Kohorte oder ein Modell je Schichtdicke -- beides bisher NICHT gemessen.
(c) Geometrische Augmentierung bleibt gestrichen, aber aus einem anderen Grund als gedacht: nicht weil sie das Netz verwirrt, sondern weil sie ein Achtel der Blutungen unsichtbar macht.
**Grenzen:** ein Lauf (Startwert 42); die Dickschicht-Stoerung trifft auch die schon dicken Kohorten 1 / 3 ein ZWEITES Mal -- die Werte sind also "zusaetzlicher Verlust an z-Aufloesung", nicht "was 4-mm-Schichten kosten".

### 5s Architektur auf sauberen Daten nachmessen (5q d; Nutzerfreigabe 21.09. abends; eingereiht ans Ende der Schlange)
**Beobachtung:** das Turnier (4.4) lief mit 40 Epochen und Fruehstopp auf dem alten, uebermalten Gitter; die Uebermalung traf 109 von 236 Blutungen (4.1). Die Rangfolge (aniso 0.404, nnU-Net-Bauart 0.265, Attention-U-Net 0.245) stammt also aus
beschaedigten Daten und kurzem Budget -- ein Gutachter wird fragen, ob sie auf sauberen Daten haelt. **Experiment:** `a02-nnunet` und `a01-attunet` im heutigen Rezept (`--epochen 60 --gitter dev/gitter_d`, Startwert 42), gepaart gegen
a03-aniso-e60-gd an der festen Zelle, mit und ohne Liquor-Regel und je Kohorte. Bewusst 60 statt 120 Epochen: fuer 60 hat die Basis drei Startwerte als Rauschmass, und 5o zeigt, dass spaete Epochen hier eher schaden.
**Vorab festgelegt:** die Basis bleibt, ausser ein Herausforderer liegt gepaart GESICHERT vorn; liegt einer innerhalb von 0.03 (Rauschboden), bekommt er einen zweiten Startwert, bevor irgendetwas geschlossen wird. Berichtet wird das Ergebnis
in jedem Fall -- auch "Rangfolge haelt" ist fuer das Paper eine Aussage. **Erwartung:** Rangfolge haelt, Abstand schrumpft (die Uebermalung hat allen Netzen geschadet, dem Sieger nicht weniger).
**Lauffaehigkeit vorab geprueft (21.09. 22:14, CPU, `turnier.py --probe --gitter dev/gitter_d`: 2 Epochen, 7 Faelle, schreibt nur nach `ergebnisse/turnier-probe/`):** a02-nnunet ok (16 544 675 Parameter), a01-attunet ok (22 667 381) -- beide
laufen im heutigen Geruest; die gemeinsame Statusdatei `dev/turnier_probe.json` wurde danach auf den Stand vor der Probe zurueckgesetzt (Kopie der CPU-Probe: `dev/turnier_probe.json.cpu-probe-0921`).
**Ergebnis a02-nnunet (22.09. 18:20; gewaehlte Zwischenstaende 50 / 55 / 15 / 55 / 50; 16.5 Mio. Parameter; `ergebnisse/analysis/valdo_nnunet_*.json`, `operating_point_valdo_nnunet.json`):** feste Zelle roh **0.458 gegen 0.585, gepaart
-0.127 [-0.202; -0.048]** (ueber dem Rauschmass 0.08); P 0.38 gegen 0.52, S 0.57 gegen 0.67, Fehlalarme je Fall 2.2 gegen 1.5; mit Liquor-Regel 0.515 gegen 0.646; kreuzweise 0.495 gegen 0.622 (-0.127 [-0.245; -0.020]). Je Kohorte -0.129 / -0.134
[-0.257; -0.027] / -0.093 -- in allen drei Kohorten hinten, gesichert in der duennschichtigen. Alle fuenf Falten liegen unter der Basis (0.525 / 0.471 / 0.483 / 0.511 / 0.432 gegen 0.645 / 0.500 / 0.629 / 0.556 / 0.577), auch unter dem
schwaechsten Basis-Startwert (0.501 gepoolt). **Gelesen:** die Rangfolge aus dem Turnier haelt auf sauberen Daten; der Abstand ist mit 0.127 kleiner als im Turnier (0.404 gegen 0.265 = 0.139), aber nicht wesentlich -- die Uebermalung hat beiden
Netzen etwa gleich geschadet, und die Erwartung "Abstand schrumpft deutlich" trifft nur knapp zu. Anders als bei allen Trainingshebeln fallen hier BEIDE Achsen (weniger Sensitivitaet UND weniger Precision): das isotrope Netz mit Pooling in
allen drei Richtungen von der ersten Stufe an verliert bei 1-mm-Schichtabstand aus 3-4-mm-Schichten die Information, die die anisotrope erste Stufe bewahrt (4.4, "zuerst nur der gemessenen Richtung trauen"). **Entscheidung nach der Regel:**
Basis bleibt; kein zweiter Startwert (Abstand > 0.03). Der zweite Herausforderer (a01-attunet) laeuft.
**Ergebnis a01-attunet (22.09. 20:27; gewaehlte Zwischenstaende 50 / 55 / 55 / 50 / 40; 22.7 Mio. Parameter; `ergebnisse/analysis/valdo_attunet_*.json`, `operating_point_valdo_attunet.json`):** feste Zelle roh **0.463 gegen 0.585, gepaart
-0.121 [-0.190; -0.061]**; P 0.35 gegen 0.52, S 0.68 gegen 0.67, Fehlalarme je Fall 3.1 gegen 1.5; mit Liquor-Regel 0.536 gegen 0.646; kreuzweise 0.548 gegen 0.622 (-0.074 [-0.160; +0.010]). Je Kohorte -0.070 / -0.073 [-0.151; -0.012] /
**-0.200 [-0.358; -0.055]** -- der Zusammenbruch liegt in der Kohorte mit den groessten Voxeln (P 0.18, 3.5 Fehlalarme je Fall). Vier der fuenf Falten unter der Basis (Ausnahme Falte 1: 0.518 gegen 0.500).
**Gelesen:** das Attention-U-Net ist KEIN schwaecherer Finder (S 0.68 = Basis), sondern ein Fehlalarm-Erzeuger -- dieselbe Achse wie alle Empfindlichkeits-Hebel (5j-5m). Die Gates gewichten Kontext, den es bei 3-4-mm-Schichten nach dem
isotropen Pooling nicht mehr gibt. **Bilanz der Gegenprobe (5s):** beide Herausforderer bleiben auf SAUBEREN Daten und im heutigen Rezept gesichert hinter der anisotropen Basis; die Rangfolge des Turniers (4.4) haelt, obwohl sie an
uebermalten Daten mit 40 Epochen und Fruehstopp entstand. Die Abstaende schrumpfen leicht (Turnier: nnU-Net -0.139, Attention -0.159; jetzt -0.127 und -0.121) -- die Uebermalung hat allen Netzen aehnlich geschadet. Damit ist die
Gutachterfrage "haelt die Architekturwahl auf sauberen Daten?" mit Ja beantwortet, ohne dass das ganze Turnier wiederholt werden musste (2 statt 10 Laeufe). **Grenzen:** ein Startwert je Herausforderer; die uebrigen acht Bauarten des
Turniers sind nicht nachgemessen -- sie lagen dort aber noch weiter hinten (0.080-0.294 gegen 0.404), und zwei von zwei nachgemessenen bestaetigen die Ordnung.
**Nutzerfrage 22.09. 20:34: "das Attention-U-Net hat die hoechste Sensitivitaet -- kann der Klassifikator daraus ein besseres F1 machen?" -- nein, und das ist ohne GPU entscheidbar.**
Fuer eine zweistufige Kette ist das F1 der ersten Stufe das falsche Mass; entscheidend ist die OBERGRENZE der Kandidatenmenge (wie viele Referenz-Blutungen bei der niedrigen Kandidatenschwelle 0.15 ueberhaupt beruehrt werden). Fehlalarme kann
die zweite Stufe verwerfen, eine nie gefundene Blutung ist endgueltig verloren. Gemessen (`cmb.stage2.candidates --keep-csf`, CPU, 4 min):
| erste Stufe | Kandidaten | davon falsch | erreichte Blutungen (von 139) | Obergrenze S | F1-Decke bei fehlerfreiem Klassifikator |
|---|---|---|---|---|---|
| Basis a03-aniso | 278 | 172 | 103 | 0.74 | 0.851 |
| a01-attunet | 407 | 297 | **104** | 0.75 | **0.856** |
| Laesions-Verlust (blob) | 370 | 252 | 110 | 0.79 | 0.884 |
| a02-nnunet | 328 | 230 | 92 | 0.66 | 0.797 |
Das Attention-Netz erreicht EINE Blutung mehr und bezahlt mit 125 zusaetzlichen falschen Kandidaten; seine hoehere Sensitivitaet am Arbeitspunkt 0.3 ist also keine zusaetzliche Information, sondern breiter gestreute Wahrscheinlichkeitsmasse --
die Basis verdaechtigt dieselben Blutungen bei 0.15 bereits. Selbst mit fehlerfreier zweiter Stufe endete die Kette bei 0.856 gegen 0.851, ein Zehntel des Rauschbodens. **Der Gegentest liegt vor (5j):** das blob-Netz hatte mit 110 erreichten
Blutungen echten Spielraum (Decke 0.884) -- die Endkette wurde trotzdem schlechter (0.594 gegen 0.625; Kette S 0.63 statt 0.60, aber P 0.56 statt 0.65), der Klassifikator reichte die zusaetzlichen Kandidaten ueberwiegend als Fehlalarme durch.
Wenn +7 erreichbare Blutungen nichts bringen, bringt +1 nichts. **Entscheidung: kein GPU-Lauf.**
**Regel fuers Geruest (dritte dieser Art heute, nach 5v und der Speichergrenze):** eine erste Stufe ist fuer eine zweistufige Kette nur dann besser, wenn sie die OBERGRENZE hebt, nicht wenn sie am Arbeitspunkt empfindlicher ist. Die Obergrenze
kostet 4 CPU-Minuten und ist VOR jedem Trainings- oder Stufe-2-Lauf zu messen. Umgekehrt gilt: die einzige Richtung, die der Kette wirklich helfen kann, ist eine erste Stufe mit hoeherer Obergrenze UND vergleichbarer Kandidatenzahl -- keine
der bisher gemessenen Varianten schafft das (blob: +7 Blutungen, aber +92 Kandidaten).

### 5u Dritter Startwert der Basis und Drei-Startwerte-Ensemble (22.09. 11:57; `ergebnisse/analysis/valdo_seeds_s2_gepoolt.json`, `operating_point_valdo_seeds_s2.json`, `valdo_s2_split.json`)
**Zweck:** Rauschmass des Rezepts auf drei Punkte stellen und pruefen, ob das Ensemble ueber zwei Mitglieder hinaus waechst (5j sagte: flach ab zwei). **Ergebnis (Startwert 2, gewaehlte Zwischenstaende 35 / 35 / 45 / 50 / 45):**
| feste Zelle | s42 | s1 | s2 | Mittel (Spanne) | Ensemble 2 (s42 + s1) | Ensemble 3 |
|---|---|---|---|---|---|---|
| roh | 0.585 | 0.553 | 0.501 | 0.546 (0.084) | 0.594 | 0.581 |
| mit Liquor-Regel | 0.646 | 0.590 | 0.578 | 0.605 (0.068) | 0.647 | 0.636 |
Gepaart gegen s42 (roh): s1 -0.032 [-0.110; +0.042], **s2 -0.083 [-0.135; -0.033]**, Ensemble 2 +0.009, Ensemble 3 -0.004. Startwert 2 je Kohorte -0.106 / -0.066 [-0.110; -0.024] / -0.093; P 0.42, 2.1 Fehlalarme je Fall (s42: 0.52, 1.5).
**Gelesen:** (1) Startwert 42 ist der beste von drei, Startwert 2 der schlechteste -- der Abstand zwischen zwei Laeufen des IDENTISCHEN Rezepts ist mit -0.083 groesser als die meisten "Effekte" dieses Dokuments und wird vom Fall-Bootstrap als
gesichert ausgewiesen: das Rauschmass in Abschnitt 3 ist entsprechend nachgetragen. (2) Ensemble: das dritte Mitglied (das schwache) senkt das Ensemble leicht (0.647 -> 0.636); Ensemble ~ bestes Mitglied, +0.03 ueber dem mittleren Mitglied --
die Aussage aus 5j haelt. Das Zwei-Startwerte-Ensemble bleibt Standard; ein drittes Mitglied lohnt nur, wenn es nicht das schwaechste ist, und das weiss man vorher nicht. (3) Wo der Startwert wirkt: Precision und Fehlalarme, nicht Sensitivitaet
(0.63-0.67 bei allen drei) -- dieselbe Achse wie bei allen Trainingshebeln. **Entscheidung:** Basis wird im Paper als 0.605 +- 0.034 (drei Startwerte, mit Liquor-Regel) berichtet; das Zwei-Startwerte-Ensemble (0.647) als ausgeliefertes Modell.
**Grenzen:** drei Startwerte sind fuer ein Rauschmass wenig; die Aussage "Spanne 0.07-0.08" kann mit mehr Startwerten noch wachsen.
**Nachtrag 23.09. 06:40 (Nutzervorgabe: 10 Startwerte; 5 von 10 fertig, `ergebnisse/analysis/operating_point_valdo_seeds_s0_s4.json`, `valdo_seeds_s3_s4.json`):** mit Liquor-Regel an der festen Zelle 0.646 / 0.590 / 0.578 / 0.603 / 0.608 --
**Mittel 0.605, Spanne 0.068**, also genau die Werte der drei Startwerte (0.605 / 0.068). Roh: 0.585 / 0.553 / 0.501 / 0.550 / 0.545; gepaart gegen Startwert 42 liegen die neuen bei -0.035 [-0.096; +0.024] und -0.039 [-0.096; +0.011].
Das Bild ist also stabil: Startwert 42 bleibt der beste von fuenf, das Mittel bewegt sich nicht, und alle vier anderen Startwerte liegen unter ihm. Die berichtete Zahl eines Modells bleibt 0.605 +- 0.034; die Precision traegt die Streuung
(0.45-0.52 roh), die Sensitivitaet nicht (0.67-0.69).

### 5w Die Decke der zweistufigen Kette -- und der erste gemessene Weg, sie zu heben (22.09. 22:09; `cmb/analysis/candidate_ceiling.py`, neu; `ergebnisse/analysis/candidate_ceiling_valdo.json`)
**Ausgangspunkt (5s):** die zweite Stufe kann Kandidaten nur verwerfen, nie eine uebersehene Blutung erfinden. Die Decke der ganzen Kette steht also im Kandidatenschritt. Bisher gemessen war nur EIN Punkt (Basis-Lauf s42, Schwelle 0.15:
103 von 139, F1-Decke 0.851 bei fehlerfreier zweiter Stufe -- erreicht sind 0.647). Zwei naheliegende Fragen waren nie gestellt: haengt die Decke an der Kandidatenschwelle, und uebersehen verschiedene Startwerte DIESELBEN Blutungen?
Beides ist ohne Training und ohne GPU aus den gespeicherten Wahrscheinlichkeitskarten zu beantworten.
| erste Stufe | Schwelle 0.15 | 0.10 | 0.05 | 0.02 |
|---|---|---|---|---|
| s42 | 103 (Decke 0.851) | 108 | 112 | 113 (0.897) |
| s1 | 99 | 103 | 103 | 103 (0.851) |
| s2 | 103 | 110 | 112 | 113 (0.897) |
| **VEREINIGUNG der drei** | **112** (0.892) | **116** (0.910) | **120** (0.927) | 121 (0.931) |
| Kandidaten der Vereinigung | 885 | 1127 | 1258 | 1316 |
**(1) Die Schwelle 0.15 verschenkt Blutungen.** Von 0.15 auf 0.05 gewinnt der Basis-Lauf 9 Blutungen (103 -> 112) fuer 128 zusaetzliche Kandidaten; danach ist es ausgereizt (0.02 bringt eine weitere). Die Schwelle war nie begruendet
gewaehlt -- sie stammt aus dem ersten Versuch am 19.09.
**(2) Verschiedene Startwerte uebersehen VERSCHIEDENE Blutungen.** Bei Schwelle 0.05 erreicht s42 sechs Blutungen, die kein anderer Lauf erreicht, s2 fuenf. Die Vereinigung kommt auf 120 von 139 (Decke 0.927) gegen 112 des besten
Einzellaufs. **Das ist genau die Information, die unser jetziges Ensemble wegwirft:** `cmb.analysis.ensemble` MITTELT die Wahrscheinlichkeitskarten, dampft also, was nur ein Mitglied sieht (5j: Ensemble ~ bestes Mitglied). Auf
Kandidaten-Ebene ist die Vereinigung das richtige Verknuepfungszeichen, nicht der Mittelwert -- weil die zweite Stufe die Fehlalarme ohnehin sortieren soll.
**Was das wert ist:** die Decke der Kette stiege von 0.851 auf 0.927; nebenbei bekaeme der gemeinsame Klassifikator statt 278 VALDO-Kandidaten (106 positiv) rund 1258 (die zweite Stufe ist auf VALDO datenarm, 5j).
**Was dagegen spricht, ehrlich:** heute gemessen (5s) hat ein empfindlicheres Netz mit hoeherer Decke (blob, 110 erreicht, Decke 0.884) die Kette VERSCHLECHTERT (0.594 gegen 0.625) -- der Klassifikator reichte die zusaetzlichen
Kandidaten als Fehlalarme durch. Eine hoehere Decke ist notwendig, nicht hinreichend. Der Unterschied hier: die zusaetzlichen Kandidaten stammen aus demselben Rezept (gleiche Fehlerstatistik) und die Trainingsmenge der zweiten Stufe
waechst um das Vierfache -- beides war beim blob-Versuch nicht der Fall.
**Vorgeschlagenes Experiment (nicht eingereiht, wartet auf den Nutzer; ~40 min GPU, keine neuen Trainings noetig):** Kandidaten aus der Vereinigung der drei vorhandenen VALDO-Startwerte bei Schwelle 0.05 bauen
(`cmb.stage2.candidates` je Lauf, dann Komponenten zusammenfuehren -- ein Werkzeug dafuer fehlt noch), gemeinsames 3D-CNN darauf (`pooled.py --models cnn`), gepaart gegen die heutige Kette. **Vorab festgelegt:** uebernommen nur, wenn
der gepaarte Gewinn ueber Null liegt UND sich mit einer unabhaengigen zweiten Kohorte wiederholt -- dieselbe Huerde wie beim CNN (5j). Erwartung: die Sensitivitaet der Kette steigt (0.60 -> 0.65-0.70), die Precision faellt; ob das F1
steigt, haengt allein daran, ob der Klassifikator die 980 zusaetzlichen Fehlalarme sortieren kann. **Fuer catalina** braeuchte es je Kohorte einen zweiten Startwert (T2* fehlt, je ~1.7 GPU-h).
**Grenzen:** die Decke ist mit der festen Nachbearbeitung (>= 1 mm3, Klumpen weg) gerechnet; 19 Blutungen bleiben auch bei Schwelle 0.02 von keinem Lauf beruehrt -- fuer die ist kein zweistufiges Verfahren zustaendig, sondern die
erste Stufe oder die Referenz (5b).

### 5x Wie viel unseres Fehlers ist die Referenz? -- drei Ersatzmessungen ohne Befunder (22.09. 22:20; `cmb/analysis/label_noise_proxies.py`, neu; `ergebnisse/analysis/label_noise_proxies_valdo.json`)
**Anlass:** die verblindete Pruefseite (5b) beantwortet die Frage sauber, braucht aber einen Kliniker. Der Nutzer hat am 22.09. klargestellt, dass er keiner ist -- meine Planung hatte das seit dem 21.09. stillschweigend angenommen, ohne
es je zu pruefen. Bis ein Befunder zur Verfuegung steht, grenzen drei objektive Ersatzmessungen die Frage ein. Zuvor gepruefte, aber untaugliche Wege: VALDO liefert nur EINE Referenzmaske je Fall (keine Befunder-Masken, am Quelldatensatz
nachgesehen); die 8 moeglichen Doppelpersonen zwischen catalina-T2* und -SWI tragen ALLE keine Blutung, aus zwei leeren Masken folgt nichts.
**Gemessen am gemeldeten Lauf (Basis-Lauf s42, feste Zelle mit Liquor-Regel: 92 Treffer, 54 Fehlalarme, 47 uebersehen, F1 0.646):**
| | von SHIVA-CMB bestaetigt | von beiden anderen Startwerten bestaetigt |
|---|---|---|
| 94 Treffer-Komponenten | 66 (70 %) | 75 (80 %) |
| **54 Fehlalarme** | **17 (31 %)** | 28 (52 %) |
| 47 uebersehene Referenz-Blutungen | 10 (21 %) | 2 (4 %) |
**(1) Ein Drittel unserer Fehlalarme findet auch ein fremdes Verfahren.** SHIVA-CMB ist auf 450 anderen Scans trainiert und kennt unsere Kette nicht. 17 von 54 Fehlalarmen markiert es unabhaengig; **16 davon finden zusaetzlich alle drei
unserer Startwerte.** Das sind 16 Orte, an denen vier unabhaengig trainierte Netze eine Blutung sehen und die Referenz nein sagt. Rechnet man jeden von SHIVA bestaetigten Fehlalarm als echte Blutung, stiege das F1 von 0.646 auf **0.722** --
das ist die Obergrenze dessen, was dieser Ersatz rechtfertigt, KEIN Schaetzwert. Zur Einordnung: SHIVA findet nur 70 % der von der Referenz bestaetigten Blutungen; waeren alle 54 Fehlalarme echt, muesste es rund 38 davon finden, es findet 17.
**(2) Die uebersehenen Blutungen sind KEIN Etikettproblem.** 37 der 47 findet weder SHIVA noch ein anderer Startwert, 38 der 47 findet kein weiterer Startwert. Unsere Sensitivitaetsluecke ist also echt und stabil -- sie liegt an den Daten
(z-Aufloesung, 5v) oder an der Methode, nicht daran, dass die Referenz zu viel markiert haette. Das schliesst die naheliegende bequeme Erklaerung aus.
**(3) Fehlalarme sehen im Mittel ANDERS aus als Treffer:** 28 Voxel [18; 44] gegen 47 [29; 79], Netz-Wahrscheinlichkeit 0.70 gegen 1.00; Ort lobar 36 / tief 11 / infratentoriell 6 gegen 48 / 33 / 13. Die Mehrheit der Fehlalarme ist also
kleiner, unsicherer und haeufiger lobar -- kein Beleg gegen die Referenz. Die 16 mehrfach bestaetigten sind die interessante Teilmenge.
**Was das fuer die Zahlen des Projekts heisst:** der moegliche Etikett-Anteil (bis zu +0.076 F1) hat dieselbe Groessenordnung wie das Startwert-Rauschen (0.08, Abschnitt 3). Beide zusammen bedeuten: **Unterschiede unter etwa 0.08 sind in
diesem Projekt grundsaetzlich nicht interpretierbar** -- weder gegen das Rauschen noch gegen die Unsicherheit der Referenz. Das stuetzt rueckwirkend jede Entscheidung dieses Projekts, die auf "nicht gesichert" hinauslief.
**Grenzen, deutlich:** SHIVA und unsere Netze sind NICHT unabhaengig im strengen Sinn -- beide lernen aus aehnlichen Bildern und koennen auf dieselben Verwechsler (Gefaessquerschnitte) ansprechen. Uebereinstimmung ist konvergente Evidenz,
kein Beweis. Die Zahl 0.722 darf nicht als "unsere wahre Leistung" berichtet werden, sondern nur als obere Grenze eines Ersatzverfahrens. **Die eigentliche Messung bleibt offen und braucht einen Kliniker** -- das Paket dafuer liegt fertig
beim Nutzer (5b).

### 5y Die Decke heben hilft nicht -- Vereinigungs-Kandidaten, gemessen (23.09. 06:23; `mb-arena/stage2/pooled_union3_0922.json`)
**Vorhersage aus 5w:** die Vereinigung der Kandidaten dreier Startwerte bei Schwelle 0.05 hebt die Decke der Kette von 0.851 auf 0.927 (120 statt 103 erreichbare Blutungen) und vervierfacht die Trainingsmenge des Klassifikators
(703 statt 278 VALDO-Kandidaten). Dort stand auch das Gegenargument: ein empfindlicheres Netz mit hoeherer Decke (blob, 0.884) hatte die Kette 2026-09-21 VERSCHLECHTERT.
**Ergebnis, gepaart gegen dieselbe zweite Stufe auf den alten Kandidaten:** VALDO **0.584 gegen 0.623 (-0.039 [-0.106; +0.027])**, catalina-T2* 0.672 gegen 0.672 (+0.000), catalina-SWI 0.676 gegen 0.670 (+0.006). Auch die einfachen
Ketten auf denselben Kandidaten fallen: nur Schwelle 0.522 gegen 0.565, Schwelle + Liquor-Regel 0.604 gegen 0.627. **Die Decke stieg um 0.076, das erreichte F1 fiel um 0.039.**
**Gelesen:** das Gegenargument hat gewonnen. Von den 17 zusaetzlich erreichbaren Blutungen holt der Klassifikator keine heraus, dafuer muss er 409 zusaetzliche falsche Kandidaten sortieren (581 statt 172) -- und die 378 Kandidaten, die
nur EIN Lauf findet, sind ueberwiegend genau das, was ein einzelner Lauf faelschlich sieht. Die Vereinigung sammelt also vor allem die Startwert-Fehler ein, nicht die Startwert-Funde. **Damit ist die Regel aus 5s praezisiert:** eine hoehere
Decke hilft nur, wenn die zusaetzlichen Kandidaten TRENNBAR sind; bei Kandidaten, die nur ein Lauf sieht, ist das gerade nicht der Fall. Ein billiger Vortest dafuer existiert: das Feld `n_runs` aus `merge_candidates` -- unter den
Kandidaten mit n_runs = 1 sind 378 Stueck und nur wenige echte Blutungen.
**Nicht gemessen, naheliegend:** die Vereinigung NUR aus Kandidaten, die mindestens zwei Laeufe finden (325 statt 703). Das haelt einen Teil des Deckengewinns und wirft die Einzelgaenger weg. Kostet 40 min GPU; die Decke dieser Teilmenge
ist ohne GPU auszurechnen und sollte VOR dem Lauf geprueft werden (Regel aus 5s).
**Nachtrag 28.09. -- gemessen (23.09., `mb-arena/stage2/pooled_union_min2_0923.json`):** Vereinigung nur aus Kandidaten, die mindestens zwei Läufe finden: CNN VALDO **0.608 gegen 0.623** (-0.015), catalina-T2* 0.672 gegen 0.672, catalina-SWI 0.677 gegen 0.670 (+0.007). Nichts über dem Rauschen der zweiten Stufe. **Der Decken-Weg ist geschlossen:** weder alle Einzelgänger noch die Mehrheits-Kandidaten bringen dem Klassifikator trennbare Zusatzfunde.
**Kontrolle:** dieselbe zweite Stufe auf den alten Kandidaten liefert 0.623 / 0.672 / 0.670 gegen 0.625 / 0.672 / 0.670 vom 21.09. -- die GPU-Streuung der zweiten Stufe betraegt hier 0.002, deutlich weniger als die 0.03 aus 5r.

### 5z Lehrer-Schueler als zweite Stufe (Nutzerwunsch 22.09.; 23.09. 06:23; `mb-arena/stage2/pooled_teacher_student_0922.json`)
**Original:** MicrobleedNet trainiert seinen Klassifikator per Wissensdestillation aus einem Lehrer-Netz. Bei uns lernte die zweite Stufe bisher aus harten 0/1-Etiketten -- eine Luecke in der Abweichungstabelle, die der Nutzer gefunden hat.
**Unsere Anpassung (`classifier.fit_predict_cnn(..., soft=, alpha=)`):** ein Lehrer, der mehr sieht als der Schueler, waere Betrug; ein Lehrer, der auf denselben Kandidaten trainiert wurde, die er unterrichtet, reicht Auswendiggelerntes
weiter. Die geschachtelte Auswertung liefert aber bereits ehrliche Lehrer-Wahrscheinlichkeiten AUSSERHALB der jeweiligen Falte: die inneren Modelle sagen die zurueckgehaltenen Trainingskandidaten vorher (bisher nur fuer die Schwellenwahl
benutzt und danach verworfen). Der Schueler lernt aus `(1 - alpha) * hartes Etikett + alpha * Lehrer`. Gedachter Nutzen: mit 106 positiven VALDO-Kandidaten zwingt das harte Ziel das Netz, Grenzfaelle als Gewissheit auswendig zu lernen.
**Pruefung (CPU):** ohne weiche Ziele bitgleich zum alten Modell; weiche Ziele gleich den harten Etiketten aendern nichts; alpha = 0 faellt exakt auf das Original zurueck; falsche Laenge wird abgelehnt.
**Ergebnis, gepaart gegen dasselbe CNN auf denselben Kandidaten:**
| alpha | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| 0.3 | +0.010 [-0.008; +0.039] | **+0.010 [+0.001; +0.018]** | +0.003 [-0.006; +0.011] |
| 0.5 | +0.009 [-0.006; +0.033] | +0.006 [-0.006; +0.015] | +0.002 [-0.009; +0.014] |
| 0.7 | +0.005 [-0.011; +0.027] | +0.005 [-0.006; +0.013] | +0.002 [-0.014; +0.019] |
**Gelesen, vorsichtig:** **alle neun Vergleiche sind positiv** -- das ist auffaellig, die Werte liegen aber zwischen +0.002 und +0.010 und damit unter dem Trainingsrauschen der zweiten Stufe (0.03, 5r). Nur T2* bei alpha = 0.3 schliesst die
Null knapp aus, und das ist einer von neun Vergleichen ohne Korrektur. Die Vorzeichen sind auch nicht unabhaengig (dieselben Falten, dieselben Kandidaten, geschachtelte alpha-Werte). **Entscheidung: noch nicht uebernommen.** Die Richtung
ist richtig, die Groesse ist nicht belegt. Belegbar waere sie mit der Methode, die der Nutzer selbst vorgeschlagen hat: 10 Startwerte je Bedingung (~6 GPU-Stunden fuer alpha = 0.3 gegen das einfache CNN). Das ist die erste Bedingung dieses
Projekts, fuer die sich dieser Aufwand lohnt -- weil sie konsistent positiv ist und einen Mechanismus hat, nicht weil eine Zahl gross aussieht.
**Nachtrag 28.09. -- die 10 Startwerte sind gerechnet (23.09., `mb-arena/stage2/ts10_0..9.json`, je ~49 min GPU, ~8 h gesamt) und erst heute ausgewertet:** alpha = 0.3 gegen dasselbe CNN, Differenz je Startwert -- VALDO **+0.001** (sd 0.009, 4/10 positiv), catalina-T2* **+0.003** (sd 0.004, 8/10 positiv), catalina-SWI **+0.002** (sd 0.005, 7/10 positiv). Absolut: CNN 0.632 / 0.671 / 0.672, Lehrer-Schüler 0.633 / 0.674 / 0.673. Die Richtung aus dem Erstlauf hält auf T2* (8 von 10), die Größe ist ≤ 0.003 -- weniger als die Startwert-Streuung der zweiten Stufe selbst (sd 0.009 / 0.003 / 0.004). **Entscheidung: nicht übernommen, abgeschlossen.** Nebenbefund, der bleibt: die Startwert-Streuung der zweiten Stufe ist mit 10 Läufen jetzt gemessen (VALDO sd 0.009, catalina 0.003-0.004) -- viel kleiner als die 0.03 aus 5r, die aus zwei Läufen mit verschiedenen Kandidatensätzen stammten. Für künftige Vergleiche der zweiten Stufe gilt das Zehn-Startwerte-Maß.

### 5aa Was gegen den Kohortenunterschied hilft -- erster Vorabtest ohne GPU (Nutzerfrage 23.09. 06:34)
**Ausgangslage (5n):** die Uebertragung scheitert an den BILDERN, nicht am Markierstil -- catalina-T2* liegt auf einem anderen Intensitaetsniveau (Hirn-Median 0.52 gegen 0.71 auf der invertierten Skala), beide catalina-Kohorten haben
schwaecheren Laesionskontrast, catalina-T2* hat 5-mm-Schichten. Naheliegender erster Verdacht: die Skala. Wir teilen durch das MAXIMUM im Hirn (Verfahren des Originals) -- ein einzelner heller Ausreisser verschiebt damit die ganze Kohorte.
Die Perzentil-Skala p99.5 (`--norm p995`) war 2026-09-20 INNERHALB von VALDO gemessen (+0.025, ein Startwert, nicht gesichert) -- also dort, wo die Skalen ohnehin zusammenpassen. Fuer die UEBERTRAGUNG war sie nie geprueft.
**Vorabtest (CPU, 3 min, je 20-25 Faelle; Kontrast hier = Differenz Blutung minus Ring auf der invertierten Skala, NICHT die relative Abdunklung aus 5n -- die drei Kontrastdefinitionen duerfen nicht verglichen werden, Abschnitt 11):**
| Kohorte | Hirn-Median heute (Skala max) | mit p99.5 | Laesionskontrast heute | mit p99.5 |
|---|---|---|---|---|
| VALDO | 0.706 | 0.726 | 0.117 | 0.124 |
| catalina-T2* | 0.522 | 0.555 | 0.095 | 0.102 |
| catalina-SWI | 0.692 | 0.739 | 0.062 | 0.066 |
**Ergebnis: die Perzentil-Skala schliesst die Luecke NICHT.** Sie hebt alle drei Kohorten um denselben kleinen Faktor (3-7 %); der Abstand catalina-T2* zu VALDO bleibt 0.52 gegen 0.73. Das Niveau ist also kein Ausreisser-Problem, sondern
liegt an der Verteilung selbst (engere HD-BET-Maske, anderer Gewebemix im Nenner). **Damit ist dieser Weg ohne GPU-Lauf erledigt** -- 3 CPU-Minuten statt 2 GPU-Stunden, dieselbe Regel wie bei der Robustheitsprobe (5v) und der
Kandidaten-Obergrenze (5s).
**Was uebrig bleibt, geordnet:** (1) Angleichen auf das HIRNNIVEAU statt auf einen Extremwert -- z-Normierung im Hirn oder Median-Angleich. Das trifft genau die gemessene Groesse (0.52 gegen 0.71). Die z-Normierung wurde 2026-09-19 als
schaedlich verworfen (-0.060), aber in einem PAKET mit Augmentierung und nur INNERHALB von VALDO, wo es nichts anzugleichen gibt. Fuer die Uebertragung ist sie ungemessen. Kosten: ein Training (~1.7 h) plus Zero-shot (~12 min).
(2) Arbeitspunkt je Kohorte, gewaehlt nach einer MESSBAREN Bildeigenschaft (Schichtdicke aus dem Dateikopf, Hirn-Median) statt nach der Leistung -- damit leckfrei. Ungemessen. (3) Histogramm-Angleich an eine Referenzverteilung vor der
Vorhersage, der klassische Weg der Domaenenanpassung; teurer als (1) und ohne Vorteil gegenueber ihm, solange (1) ungemessen ist. **Nicht machbar:** der Laesionskontrast selbst und die Schichtdicke sind Physik.
**Alle Kandidaten gemessen (23.09. 08:02; `cmb/analysis/harmonisation_check.py`, neu; `ergebnisse/analysis/harmonisation_check_0923.json`; 12 Faelle je Kohorte, CPU 3 min).** Berichtet wird die SPANNE zwischen den drei Kohorten
nach der jeweiligen Normierung -- klein heisst angeglichen. Die entscheidende Spalte ist der Laesionskontrast: das ist, was das Netz erkennt.
| Normierung | Spanne Hirn-Niveau | Spanne Hirn-Streuung | Spanne Laesionsniveau | **Spanne Kontrast** |
|---|---|---|---|---|
| max (heute) | 0.188 | 0.038 | 0.199 | 0.060 |
| p99.5 | 0.194 | 0.039 | 0.182 | 0.057 |
| z im Hirn | 0.338 | 0.186 | 0.331 | **0.362** |
| Median / IQR | 0.000 | 0.000 | 0.151 | 0.194 |
| weisse Substanz als Anker | 0.093 | 0.099 | **0.017** | 0.085 |
| **Landmarken (Nyul-Udupa)** | **0.000** | **0.000** | 0.080 | **0.018** |
**Zwei Befunde, einer davon gegen meine Erwartung.** (1) **Der Landmarken-Angleich gewinnt klar:** Niveau und Streuung exakt angeglichen (bauartbedingt) UND die Kontrast-Spanne faellt von 0.060 auf 0.018, also um den Faktor drei. Er ist
das einzige Verfahren hier, das einen NICHTLINEAREN Unterschied beheben kann -- und offenbar ist der Unterschied nichtlinear. (2) **Die z-Normierung ist die SCHLECHTESTE von sechs** (Kontrast-Spanne 0.362, sechsmal schlechter als heute),
und Median/IQR ist ebenfalls schlechter als gar nichts (0.194). Der Grund ist dieselbe Mechanik: die Streuung im Hirn wird von kohortenspezifischer Struktur und Rauschen bestimmt; wer durch sie teilt, gleicht die VERTEILUNG an und
verzerrt dabei den LAESIONSKONTRAST je Kohorte unterschiedlich. Das erklaert rueckwirkend den Befund vom 19.09. (-0.060 im Paket): die z-Normierung war dort vermutlich nicht unschuldig, sondern der schaedliche Teil.
**Folge fuer die eingereihten Laeufe:** [193]-[195] (z-Normierung, Training + zwei Zero-shot) bleiben eingereiht, aber mit geaenderter Rolle -- **nicht mehr als aussichtsreicher Kandidat, sondern als Falsifikationstest dieser Vorpruefung.**
Sagt die Vorpruefung "z ist das schlechteste von sechs" und bestaetigt der GPU-Lauf das, ist das Werkzeug fuer kuenftige Entscheidungen geeicht (2 GPU-Stunden dafuer sind gut angelegt). Sagt der Lauf das Gegenteil, misst die Vorpruefung
die falsche Groesse und muss ueberdacht werden. **Der eigentliche Kandidat ist jetzt der Landmarken-Angleich** -- der braucht noch Code in `turnier.py` (`--norm landmarks` mit hinterlegten Referenz-Landmarken) und im Zero-shot.
**Zwei 0-1-Varianten nachgemessen (Nutzerfrage 23.09. 11:01; `ergebnisse/analysis/harmonisation_check_0923b.json`):** `minmax` (dunkelstes Hirnvoxel auf 0, hellstes auf 1) liefert Spanne 0.188 / 0.038 / 0.199 / **0.060** -- ZIFFERNGLEICH
mit der heutigen Skala, weil diese bauartbedingt schon min-max ist (siehe Abschnitt 11). Die robuste Fassung `p1_p99` (1. und 99. Perzentil statt der Extreme, auf [0,1] gekappt): 0.171 / 0.038 / 0.161 / **0.063**, also beim Kontrast
minimal schlechter als heute. **Damit ist die Rangliste vollstaendig:** Landmarken 0.018 | p99.5 0.057 | max = min-max 0.060 | 0-1 robust 0.063 | weisse Substanz 0.085 | Median/IQR 0.194 | z 0.362. **Alle AFFINEN Skalen landen
zwischen 0.057 und 0.063** -- das ist die Grenze dessen, was eine lineare Abbildung hier leisten kann, und sie wird nur vom stueckweise linearen Landmarken-Angleich gebrochen (Faktor drei). Das ist zugleich das staerkste Argument dafuer,
dass der Kohortenunterschied NICHTLINEAR ist.
**Grenze:** die Vorpruefung misst Bildstatistik, nicht Netzleistung. Sie kann nur ausschliessen und ordnen, nicht beweisen -- genau deshalb der Falsifikationstest.
**Nutzerfrage 23.09. 11:58: "koennen wir alle Versuche mit normalisierten Eingaengen wiederholen?" -- die Vorpruefung sagt: nur die kohortenuebergreifenden, und der Rest wuerde SCHADEN.** Dieselbe Messung, diesmal mit den DREI
VALDO-Teilkohorten als "Kohorten" (`ergebnisse/analysis/harmonisation_innerhalb_valdo_0923.json`, Referenz valdo-2):
| innerhalb VALDO | Spanne Niveau | Spanne Streuung | Spanne Laesionskontrast |
|---|---|---|---|
| heute (Min-Max) | 0.048 | 0.165 | **0.039** |
| Landmarken | 0.000 | 0.000 | **0.127** |
Je Teilkohorte faellt der Laesionskontrast von 0.113 / 0.104 / 0.074 auf 0.149 / 0.098 / **0.023** -- die dickschichtige Kohorte 3 verliert zwei Drittel ihres Kontrasts.
**Die Erklaerung, und daraus wird eine Regel:** die drei VALDO-Teilkohorten liegen im NIVEAU schon zusammen (0.675 / 0.722 / 0.705), unterscheiden sich aber stark in der STREUUNG (0.086 / 0.143 / 0.250). Der Landmarken-Angleich erzwingt
gleiche Perzentile und streckt damit die breit gestreute Kohorte 3 zusammen -- ihre Blutungen verlieren dabei den Kontrast, den sie gegen ihre eigene Umgebung hatten. **Landmarken helfen, wo Kohorten sich im NIVEAU unterscheiden
(VALDO gegen catalina: 0.71 gegen 0.52), und schaden, wo sie sich in der STREUUNG unterscheiden (innerhalb VALDO).**
**Vorhersage, festgehalten VOR dem Ergebnis des laufenden Laufs:** der Landmarken-Lauf wird innerhalb VALDO SCHLECHTER als 0.646 sein, und zwar am staerksten in Kohorte 3 (3-4 mm, heute 0.479). Trifft das ein, ist die Vorpruefung zum
zweiten Mal bestaetigt und darf kuenftig Laeufe ausschliessen. Beim Zero-shot auf catalina erwarte ich dagegen weiter einen Gewinn -- dort wirkt sie auf das Niveau.
**Damit die Antwort auf die Nutzerfrage:** alle 35 Schritte der Entscheidungsleiter zu wiederholen waere ueber 50 GPU-Stunden, und die Vorpruefung sagt, dass die Eingaben innerhalb VALDO dabei SCHLECHTER angeglichen waeren als heute --
die zwoelf negativen Trainingsbefunde wuerden also nicht fairer, sondern unfairer wiederholt. Sinnvoll sind nur die KOHORTENUEBERGREIFENDEN Versuche (Zero-shot, Mischen, Feintuning, mix-alle), und auch die erst, wenn der laufende Lauf
einen Gewinn zeigt. Die Reihenfolge steht damit fest: erst das Ergebnis, dann die Entscheidung -- nicht umgekehrt.
**Nutzerhinweis 23.09. 11:10: "besonders beim Mischen von Daten koennte Min-Max helfen."** Min-Max selbst ist hier eine Nulloperation -- nachgemessen auf ALLEN DREI Gittern (Minimum im Hirn 0 bis 1.8e-5, Maximum 0.9955; eine
Min-Max-Streckung waere Faktor 1.004). Der Gedanke dahinter trifft aber einen ungemessenen Punkt: **alle unsere Mischversuche liefen auf der unharmonisierten Skala.** Gemischt bei gleichem Budget je Fall ist heute gleichauf mit eigenem
Training (5h, Schritt 25) -- und genau beim Mischen sieht das Netz beide Verteilungen WAEHREND des Trainings, dort sollte eine Angleichung am meisten bringen. **Eingereiht:** gemischt VALDO + catalina-T2*, 120 Epochen, mit
Landmarken-Angleich, gegen den vorhandenen Mischlauf auf der alten Skala (catalina-T2* 0.641, VALDO-Anteil 0.582). **Vorab festgelegt:** ein Gewinn zaehlt erst ueber dem Startwert-Rauschen von 0.08 (Abschnitt 3); erwartet wird eine
Verbesserung auf BEIDEN Anteilen, denn die Angleichung nimmt dem Netz die Aufgabe ab, zwei Skalen gleichzeitig zu bedienen.
**Dafuer musste `turnier.py` geaendert werden (Sicherung .vor-landmarken-0923):** `--manifest` und `--norm` schlossen sich bisher aus, weil der Manifest-Lader den Normierungs-Lader still ueberschrieb -- die Sperre war ein Symptom, nicht
die Ursache. Jetzt wird die Normierung ZULETZT umgelegt und umschliesst den tatsaechlich benutzten Lader. CPU-Probe mit beiden Kohorten zugleich: alle vier Faelle treffen die Referenz-Perzentile auf sechs Stellen.
**Gebaut (23.09. 08:58):** `turnier.py --norm landmarken` (Sicherung .vor-landmarken-0923), `cmb.transfer.zero_shot --norm landmarken`, und `code/landmarken_bauen.py`, das die Referenz als Datei erzeugt -- eine Datei, damit Training und
Vorhersage nachweislich dieselbe Skala benutzen, auch fuer eine Kohorte, die spaeter dazukommt. Referenz aus den 57 CV-Faellen: P10 0.6278, P25 0.6619, P50 0.7064, P75 0.7939, P90 0.9514, P99 0.9766 (`dev/landmarken_valdo.json`).
**Offengelegt:** in diese Referenz fliesst die Intensitaetsverteilung jedes CV-Falles zu 1/57 ein, also auch die des jeweiligen Testfalles. Es sind reine Perzentile ohne Etiketten -- der uebliche Weg in der Literatur und ein sehr kleiner
Einfluss, aber ein Einfluss; `landmarken_bauen.py --faelle` erlaubt eine leckfreie Fassung je Falte, falls ein Gutachter darauf besteht. **Pruefung (CPU, drei Faelle):** nach der Angleichung treffen alle Faelle die Referenz-Perzentile auf
fuenf Stellen genau (vorher lagen sie bei P10 0.33-0.39 statt 0.63), der Hintergrund bleibt exakt 0, kein Hirnvoxel wird 0, das Label ist unberuehrt. **Eingereiht:** Training mit Landmarken plus Zero-shot auf beide catalina-Kohorten;
Bezug sind die vorhandenen Zero-shot-Werte (T2* 0.622, SWI 0.305) und als Rauschmass der Zero-shot der Basis mit Startwert 1 (0.638 / 0.311, ARBEIT.md 5p). **Vorab festgelegt:** die Angleichung gilt als Gewinn, wenn sie auf mindestens
einer catalina-Kohorte gepaart ueber Null liegt UND groesser ist als der Abstand zwischen den beiden Basis-Startwerten (0.016 / 0.006); innerhalb VALDO wird Gleichstand erwartet, denn dort gibt es nichts anzugleichen.

### 5ab Vortrainierter Encoder fuer die zweite Stufe (Nutzerwunsch 22.09.; Ergebnis 23.09. 09:45; `mb-arena/stage2/pooled_vortraining_0923.json`)
**Idee (Literaturliste Punkt 5, auf eigene Daten uebertragen):** die zweite Stufe hat zu wenige Etiketten (106 positive VALDO-Kandidaten), aber genug Bilder -- jede Kohorte liefert Ausschnitte, und fuers Vortraining braucht keiner davon
ein Etikett. Statt fremde Gewichte herunterzuladen (Architektur passt nicht, Herkunft ungeklaert) lernt der Encoder ZUERST, ausgeschnittene Quader im eigenen Ausschnitt zu rekonstruieren, und wird dann auf die Klassifikation feinabgestimmt.
**Warum gerade Luecken fuellen:** es ist die eine Aufgabe, die genau das lehrt, was unsere beiden Klassen trennt -- ein Gefaess SETZT SICH FORT, eine Blutung hoert auf. Wer eine Luecke fuellen will, muss lernen, wie oertliche Struktur
weitergeht. Intensitaets-Aufgaben wurden bewusst weggelassen: die Robustheitsprobe (5v) zeigt, dass das Netz gegen Gamma, Kontrast und Rauschen bereits invariant ist -- dort gibt es nichts zu lernen.
**Leckfrei:** das Vortraining sieht nur `train_x`, also die Trainingsfalten; aus der Testfalte kommt nichts hinein, auch nicht ohne Etikett.
**Vorpruefung (CPU, echte Kandidaten-Ausschnitte):** der Rekonstruktionsfehler auf UNGESEHENEN Ausschnitten faellt von 0.332 (zufaellige Gewichte) auf 0.0049 nach 12 Epochen; die triviale Messlatte (ueberall den Mittelwert raten) liegt
bei 0.105. Der Encoder lernt also echte Struktur, nicht nur das Helligkeitsniveau -- die Vorübung funktioniert.
**Ergebnis, gepaart gegen dasselbe CNN auf denselben Kandidaten:**
| Variante | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| 60 Epochen Vortraining | +0.012 [-0.009; +0.040] | -0.005 [-0.015; +0.005] | -0.011 [-0.027; +0.003] |
| 20 Epochen Vortraining | -0.000 [-0.052; +0.050] | -0.005 [-0.016; +0.005] | -0.009 [-0.029; +0.011] |
**Gelesen: kein Gewinn, und das Vorzeichen wechselt je Kohorte.** Auf VALDO (der datenaermsten Kohorte mit 278 Kandidaten) +0.012, auf beiden catalina-Kohorten leicht negativ -- kein Intervall verlaesst die Null, alle Werte liegen weit
unter dem Trainingsrauschen der zweiten Stufe (0.03, 5r). **Interpretiert:** die Vorübung lernt zwar messbar etwas (Rekonstruktionsfehler 68-fach besser als zufaellig), aber offenbar nichts, was die Trennung Blutung/Verwechsler
verbessert -- Struktur fortsetzen zu koennen heisst nicht, Gefaess von Blutung unterscheiden zu koennen. Dazu kommt ein bezahlter Preis: die drei Mitglieder starten jetzt vom SELBEN vortrainierten Encoder und unterscheiden sich nur noch
in Kopf-Initialisierung und Datenreihenfolge -- das Ensemble wird weniger vielfaeltig. Das duerfte den kleinen VALDO-Gewinn auf den beiden anderen Kohorten wieder auffressen. **Entscheidung: nicht uebernommen.**
**Grenzen:** eine Pretext-Aufgabe, eine Architektur, ein Startwert je Variante; getestet wurde Rekonstruktion, nicht Kontrastlernen (SimCLR-artig) oder Rotationsvorhersage. Die Aussage lautet: DIESE Vorübung bringt hier nichts -- nicht,
dass Vortraining grundsaetzlich nichts bringt.

### 5ac Intensitätsangleich gemessen -- Landmarken schaden überall, z-Norm ist neutral, die Vorprüfung aus 5aa hat sich NICHT bewährt (Läufe 23./24.09., ausgewertet 28.09.; `ergebnisse/analysis/operating_point_valdo_norm_0928.json`, `valdo_norm_split_0928.json`, `mb-arena/transfer/auswertung_zeroshot_{t2star,swi}_norm_0928.json`, `auswertung_t2star_mix_e120_nlm_0928.json`)
**Warum erst jetzt:** die vier Läufe (Training mit `--norm landmarken` / `--norm z`, je zwei Zero-shots, gemischt e120 mit Landmarken) waren am 24.09. 03:49 fertig, die Auswertung fehlte -- seit dem 25.09. scheitert `claude -p` in der Morgenkontrolle (rc=1, Anmeldung), niemand hat die Ergebnisse gelesen. Nachgeholt am 28.09. auf der CPU (vier Auswertungen, ~20 min).
**Ergebnis, gegen die vorab festgelegten Regeln aus 5aa gelesen (Zero-shot-Gewinn zählt nur > 0.016 / 0.006 = Abstand der Basis-Startwerte s42/s1; innerhalb VALDO Gleichstand erwartet):**
| Skala | VALDO feste Zelle (Liquor-Regel) | VALDO kreuzweise, gepaart gegen Basis | catalina-T2* zero-shot, gepaart gegen s42 | catalina-SWI zero-shot, gepaart gegen s42 |
|---|---|---|---|---|
| heute (max), s42 | 0.646 | 0.622 | 0.622 (P 0.60 S 0.65 FP 4.76) | 0.305 (P 0.32 S 0.29 FP 6.72) |
| Rauschmaß: Basis s1 | 0.590 | 0.564 | 0.638, +0.016 [+0.003; +0.033] | 0.311, +0.006 [-0.015; +0.030] |
| **Landmarken (Nyul-Udupa)** | **0.549** | 0.559, **-0.064 [-0.143; +0.013]** | **0.574, -0.048 [-0.068; -0.027]** | **0.271, -0.034 [-0.058; -0.012]** |
| z im Hirn | 0.615 | 0.618, -0.005 [-0.068; +0.066] | 0.618, -0.004 [-0.035; +0.031] | 0.283, -0.022 [-0.050; +0.006] |
Je VALDO-Teilkohorte (roh, `--split-valdo`): Landmarken **-0.194 [-0.337; -0.010]** (Kohorte 1, 4 mm: 0.557 -> 0.364), **-0.078 [-0.155; -0.016]** (Kohorte 2, 0.8 mm), **-0.138 [-0.253; -0.016]** (Kohorte 3, 3-4 mm: 0.479 -> 0.341); Fehlalarme je Fall 1.5 -> 3.9, 1.7 -> 2.7, 1.2 -> 2.1. z-Norm: -0.123 n. s. / -0.002 / +0.021, alle Intervalle um Null.
Gemischt VALDO + catalina-T2*, 120 Ep., MIT Landmarken gegen den alten Mischlauf: catalina-T2* **0.630 gegen 0.641** (gegen eigenes Training 0.663: -0.032 [-0.057; -0.009], gesichert), VALDO-Anteil **0.355 gegen 0.582** -- die Angleichung hat dem Netz beim Mischen nicht die Arbeit abgenommen, sie hat den VALDO-Anteil ruiniert.
**Gelesen, in drei Sätzen.** (1) Kein einziger der vier vorab festgelegten Vergleiche ist ein Gewinn; Landmarken sind auf beiden catalina-Kohorten und in allen drei VALDO-Teilkohorten GESICHERT schlechter, mit doppelt bis dreifach so vielen Fehlalarmen. (2) Die Vorhersage aus 5aa traf nur zur Hälfte: innerhalb VALDO schlechter -- ja (aber am stärksten in Kohorte 1, nicht 3); Zero-shot-Gewinn auf catalina -- **nein, das Gegenteil.** (3) Die z-Normierung, die die Vorprüfung als SCHLECHTESTE von sechs Skalen einstufte (Kontrast-Spanne 0.362 gegen 0.060), ist in Wahrheit überall neutral; die Landmarken, die sie als BESTE einstufte (0.018), sind überall gesichert schlechter. **Die Rangfolge der Vorprüfung ist damit falsifiziert.**
**Was das für die Werkzeuge heißt (wichtiger als das Einzelergebnis):** `cmb/analysis/harmonisation_check.py` misst die Spanne des MITTLEREN Läsionskontrasts ZWISCHEN Kohorten. Das ist offenbar nicht die Größe, an der das Netz hängt: das Netz sieht je Fall und je Läsion den Kontrast zur unmittelbaren Umgebung, und ein stückweise linearer Angleich mit Knoten an P10..P99 streckt oder staucht den Schwanz unter P10 -- dort sitzen die Blutungen -- JE FALL verschieden. Gleiche Kohorten-Mittel bei je Fall verschobenem Kontrast ist genau das, was ein Netz mehr Fehlalarme kostet. Das Werkzeug beschreibt Bildstatistik richtig (das war geprüft) und schließt trotzdem falsch auf die Netzleistung. **Regel zurückgenommen:** die Vorprüfung darf KEINE Läufe ausschließen und keine einreihen; sie bleibt als Beschreibung. Die anderen beiden CPU-Vorprüfungen (Robustheitsprobe 5v, Kandidaten-Obergrenze 5s/5w) sind je einmal bestätigt und einmal (5y) an ihrer Grenze gescheitert -- auch sie sind Ordnungshilfen, keine Entscheider. Die Lehre steht unter [[fehler-heisst-geruest]]: der Falsifikationstest hat genau das geleistet, wofür er eingereiht war, nur mit dem anderen Vorzeichen.
**Damit ist der Weg Intensitätsskala GESCHLOSSEN:** alle affinen Skalen sind gleichwertig (5aa, p99.5 innerhalb VALDO +0.025 n. s.), die zwei nicht-affinen bzw. verteilungsangleichenden verlieren oder bleiben gleich. Der Kohortenunterschied VALDO/catalina liegt nicht in der Skala. Was bleibt, ist Physik (5-mm-Schichten, schwächerer Kontrast) und der Markierstil -- und für beides ist eigenes Training je Kohorte (0.663 / 0.670 mit zweiter Stufe) die gemessen beste Antwort.
**Nicht übernommen:** `--norm landmarken`, `--norm z`. Beide Schalter bleiben im Code, `dev/landmarken_valdo.json` bleibt als Beleg liegen.

### 5ad Einfrieren und die EINMALIGE Messung auf den 15 beiseitegelegten Fällen (Nutzerentscheid 28.09. ~15:00; Vorabfestlegung VOR dem Lauf)
**Eingefrorene Kette für VALDO (nichts davon wird nach dieser Messung noch verändert):** erste Stufe a03-aniso, Rezept r3 auf `gitter_d`, 60 Epochen, Zwischenstandswahl auf der inneren Falte, Startwerte 42 und 1 (`ergebnisse/turnier/a03-aniso-e60-gd`, `a03-aniso-s1-e60-gd`); je Startwert das Mittel der fünf Faltenmodelle (so wird ausgeliefert, `cmb.transfer.zero_shot`), dann das Mittel der beiden Startwerte (`cmb.analysis.ensemble`); Nachbearbeitung feste Zelle Schwelle 0.3 / mindestens 2 mm3 / Riesenkomponenten weg; Liquor-Regel (Mehrheitslabel SynthSeg = freier Liquor oder Ventrikel -> verwerfen). Die zweite Stufe (gemeinsames 3D-CNN) gehört zur Kette für catalina-SWI; auf VALDO war sie gleichauf mit der Liquor-Regel (5e, 0.623 gegen 0.627) und wird hier NICHT angewandt -- die Messung gilt der VALDO-Auslieferung, wie sie in der CV gemessen wurde (Zwei-Startwerte-Ensemble mit Liquor-Regel 0.647, Abschnitt 5u).
**Die 15 Fälle:** `dev/manifest_valdo_gitter_d_heldout.json` (neuer Schalter `build_valdo_manifest --held-out`; das CV-Manifest wurde dabei nachgebaut und ist bis auf die Reihenfolge identisch; die 15 Fälle tragen im Manifest die HILFS-Falte 0, weil alle Lader nur Falten 0-4 annehmen -- erster Startversuch scheiterte daran mit KeyError; die Falte ist nur ein Ordnername, trainiert wird nichts). 97 Referenzblutungen, davon **73 in einem Fall (sub-110, Kohorte 1)**, 11 in sub-221, drei Fälle ohne Blutung, neun Fälle mit 1-3. Diese Fälle wurden am 14.09. schon EINMAL berührt (we5, altes Modell r3 mit Mehrheitsvotum: F1 0.19; Abschnitt 6, 10e) -- das hier ist die zweite und letzte Berührung. Kein Modell, keine Schwelle, keine Regel wurde seit dem 14.09. an diesen Fällen gewählt.
**Erwartung, festgehalten vor dem Ergebnis:** gepooltes F1 mit Liquor-Regel zwischen 0.55 und 0.70. Begründung: die CV-Zahl ist 0.647 (Ensemble) bzw. 0.605 ± 0.034 (Einzelstartwerte), und die Klasse ">5 Blutungen" lag in der CV bei 0.72 -- sub-110 trägt drei Viertel der Referenz, das gepoolte F1 ist also im Wesentlichen das F1 dieses einen Falls. Deshalb werden VORAB drei weitere Zahlen festgelegt und berichtet: (a) gepoolt ohne sub-110 (24 Blutungen, 14 Fälle), (b) Median des F1 je Fall (Challenge-Regel, `subject_level_f1`), (c) Fehlalarme je Fall in den drei blutungsfreien Fällen. Eine Abweichung nach unten unter 0.50 würde ich als Überanpassung der Kette an die CV lesen (viele Vergleiche auf denselben 57 Fällen, Abschnitt 6); eine nach oben über 0.70 als Losgewinn durch sub-110, nicht als Beleg.
**Was NICHT passiert:** kein zweiter Versuch, keine Zelle wird nach dem Blick auf diese Zahl gewählt, kein Startwert nachgeschoben. Das Ergebnis geht so ins Paper, mit Intervall.
**Ausführung:** `cmb.transfer.zero_shot` je Startwert (GPU über `grossauftrag --vram 20`), `cmb.analysis.ensemble`, dann `cmb.transfer.evaluate` (roh), `cmb.analysis.operating_point` (nur die feste Zelle mit Liquor-Regel zählt; die kreuzweise Wahl ist mit einer einzigen Falte bedeutungslos) und `cmb.analysis.subject_level_f1`. Ergebnis in `ergebnisse/validierung/heldout_0928/`.
**ERGEBNIS (28.09. 16:10, GPU 377 s für beide Startwerte, Auswertung CPU; `ergebnisse/validierung/heldout_0928/summary.json`, Modell-Prüfsummen `FROZEN.sha256`):**
| Messung | Ensemble s42+s1 | s42 | s1 |
|---|---|---|---|
| gepoolt, 15 Fälle, 97 CMB, feste Zelle + Liquor-Regel | **0.468** | 0.472 | 0.468 |
| gepoolt roh (P / S / FP je Fall) | 0.463 (0.68 / 0.35 / 1.07) | 0.480 | 0.456 |
| (a) gepoolt OHNE sub-110, 14 Fälle, 24 CMB, mit Liquor-Regel | **0.731** (roh 0.679, P 0.59 S 0.79) | 0.704 | 0.731 |
| (b) Challenge-Regel, Median F1 je Fall (12 Fälle mit Blutung), mit Liquor-Regel | **0.667** (Mittel 0.591; Mittel über alle 15: 0.540) | 0.667 | 0.667 |
| (c) blutungsfreie Fälle (3) | 1.0 Fehlalarm je Fall roh, 0.73 mit Regel; 1 von 3 ganz ohne Fehlalarm | | |
| nur sub-110, 73 CMB | roh F1 0.330, P 0.83, **S 0.20**, 3 Fehlalarme; mit Regel 0.315 | | |
**Gelesen, gegen die Vorabfestlegung.** Das gepoolte F1 liegt mit 0.468 UNTER der vorab genannten Spanne 0.55-0.70 -- und die vorab festgelegte Zahl (a) erklärt vollständig, warum: ohne sub-110 liegt die Kette bei 0.731, also am OBEREN Rand der CV-Zahlen (0.647 Ensemble, 0.605 ± 0.034 Einzelstartwerte). Der Median je Fall nach Challenge-Regel (0.667) liegt über dem CV-Wert derselben Regel (0.56 mit Liquor-Regel, Abschnitt 5c/5a) und in der Nähe der Challenge-Spitze (0.67, drei Sequenzen). Die Kette ist also NICHT an die CV überangepasst -- auf 14 der 15 Fälle hält sie, was die CV versprach, und mehr.
**Was sub-110 ist:** der schwerste Fall des gesamten Datensatzes (73 Blutungen; die drei schwersten CV-Fälle der Kohorte 1 haben 12 / 6 / 6, im ganzen CV-Satz hat kein Fall mehr als ~30). Seine Blutungen sind klein (Median 10 mm3, 35 von 73 unter 10 mm3) und kontrastarm: von den 36 Läsionen unter dem Kontrast-Median findet das Netz 1 (Sensitivität 0.03), von den 37 darüber 14 (0.38); 34 der 46 verpassten lobären Läsionen tragen eine Höchstwahrscheinlichkeit unter 0.05 -- sie werden nicht knapp verfehlt, sondern gar nicht gesehen (`missed_lesions_nur110.json`). 12 Läsionen liegen nach SynthSeg im Liquor-Label (Sensitivität dort 0.08). Precision auf diesem Fall ist 0.83: was das Netz meldet, stimmt. Das ist genau das Bild aus 5c/5i für kleine, kontrastarme Blutungen -- hier in einem Fall 73-fach.
**Wie das ins Paper geht (vorab festgelegt: so, mit Intervall, kein zweiter Versuch):** beide Zahlen nebeneinander -- gepoolt 0.468 über 15 Fälle, davon ein Fall mit 75 % der Referenz und Sensitivität 0.20; ohne ihn 0.731; Median je Fall 0.667. Die ehrliche Aussage: die Kette hält auf gewöhnlichen Fällen (0-11 Blutungen), und sie versagt bei massiver Last kleiner, kontrastarmer Blutungen, wie sie im Training nicht vorkam (Kohorte 1 hat im CV-Satz nur drei Fälle über 5). Die Sensitivität für solche Fälle ist die offene Grenze des Verfahrens, nicht ein Messfehler. **Damit ist die VALDO-Kette eingefroren und gemessen.** Nachtrag Abschnitt 6 und 10e.

### 5ae Forschungslinie nach dem Einfrieren: drei Arme, vorab festgelegt (Nutzerauftrag 28.09. ~17:00 "ein Research starten"; Recherche `recherche/moeglichkeiten-0928.md`)
**Ausgangslage:** die VALDO-Kette ist eingefroren und gemessen (5ad); die 15 Fälle sind verbraucht. Alles Weitere wird NUR in der 5-fach-CV auf den 57 Fällen gemessen, gepaart, mit zwei Startwerten (Rauschmaß 0.08 für Einzelstartwerte, Abschnitt 3). Maßstab: andere Information oder andere Trainingsverteilung, nicht mehr Empfindlichkeit (5q, 5ac).
**Arm 1 -- künstliche Blutungen und Verwechsler in der ERSTEN Stufe** (CenSynCMB 2026: auf VALDO 71.7 -> 74.3 Läsions-F1 als letzter Schritt ihrer Kette; unsere Lücke: das Netz hat fast keine Beispiele für massive Last kleiner, kontrastarmer Blutungen, 5ad).
Umsetzung `cmb/synthesis/build_synthetic_grid.py` -> `dev/gitter_s`: je CV-Fall werden N ~ U{2..6} künstliche Blutungen und M ~ U{3..8} Gefäß-Verwechsler ins Bild von `gitter_d` gesetzt, FRST wird wie im Gitterbau NEU gerechnet, das Label um die künstlichen Blutungen erweitert (Verwechsler bleiben 0). Alle Parameter kommen aus der GEMESSENEN Verteilung der 139 echten Blutungen (`ergebnisse/analysis/lesion_stats_valdo_0928.json`, `cmb/synthesis/lesion_stats.py`): Volumen je Kohorte per Bootstrap aus den echten Werten bis zum 75. Perzentil (Median 10 mm3; Ko1 12.5, Ko2 7.3, Ko3 16), Spitzenkontrast je Kohorte aus der echten Verteilung (Median 0.20; Ko3 0.12), Gewebeort nach den echten Anteilen (Rinde 29 %, Mark 30 %, tiefe Kerne 27 %, infratentoriell 11 %; kein Liquor), mindestens 6 mm Abstand zu echten und anderen künstlichen Läsionen, 3 mm zum Hirnrand. Form: Gauß-Ellipsoid (Halbwerts-Volumen = gezogenes Volumen, Anisotropie in der Schicht 0.8-1.25). Schichtdicke: das Ellipsoid wird je Quellschicht-Block (4 mm Ko1, 0.8 mm Ko2, 3-4 mm Ko3, aus `info.json` des Falls) in z gemittelt und linear zurückinterpoliert -- dieselbe Mechanik wie Aufnahme + Gitterbau; das Label füllt wie beim Nächste-Nachbar-Gitterbau den ganzen Block. Nach der Mittelung wird der Spitzenkontrast auf den gezogenen Wert skaliert, damit die Kontrastverteilung NACH der Verarbeitung der echten entspricht. Verwechsler: Gauß-Röhren (zufällige Richtung, Länge 6-16 mm, Halbwertsbreite 0.8-1.6 mm, Kontrast 0.7-1.0 des gezogenen), Ort Rinde/Mark. Additiv auf der invertierten Skala, auf [0, 1] gekappt, Hirnmaske unverändert.
Training: neuer Schalter `turnier.py --synthese dev/gitter_s` (Ordnerzusatz `-syn`): ein Stichproben-Wrapper wie `mit_harten_negativen` tauscht NUR für die Trainingsfälle der Falte Bild/FRST/Label gegen die künstliche Fassung, zieht die Stichprobe und stellt die echten Daten wieder her. Test- und innere Wahlfalte sehen nie ein künstliches Voxel; ohne Schalter ist der Code unberührt. Nebenwirkung, offen gelegt: die Läsions-Stichprobe (60 %) zieht gleichverteilt über echte UND künstliche Komponenten -- bei ~4 künstlichen gegen 2.4 echten je Fall sind etwa 60 % der Läsionsausschnitte künstlich.
**Arm 2 -- kubische Interpolation des Bildes** (4.8, offen seit 19.09.): `dev/gitter_k` = `gitter_c_bauen.py` mit `GITTER_C_ORDNUNG=3` aus `preproc_roh` (die lineare Fassung wurde vorher an sub-201 bitgleich zu `gitter_d` nachgebaut), Label und Hirnmaske identisch zu `gitter_d`, Bild auf [1e-6, 1] gekappt. Die Umrechnung kostet linear 15-30 % Läsionskontrast; kubisch sollte davon einen Teil zurückholen. Nach 5ac wird KEINE CPU-Kontrastmessung vorgeschaltet, die den Lauf entscheidet -- sie wird nur zur Beschreibung mitgeführt.
**Arm 3 -- Momeni/CSIRO-SWI-Datensatz** (75 SWI, 175 echte + 750 künstliche Blutungen, nur Koordinaten, CSIRO Data Licence, frei: doi 10.25919/AEGY-NY12, data.csiro.au/collection/csiro:50304): Lizenz lesen, herunterladen, Bildparameter prüfen; Ziel = erster UNABHÄNGIGER öffentlicher Test des SWI-Modells (Zentrumsabstand-Metrik, weil keine Masken) und ggf. Trainingszuwachs für SWI. Kein GPU-Aufwand vor der Sichtung.
**Läufe (eingereiht, je ~1.1 h GPU):** Arm 1: a03-aniso e60 gitter_d + Synthese, Startwerte 42 und 1. Arm 2: a03-aniso e60 gitter_k, Startwerte 42 und 1. Bezug: Basis gitter_d s42 0.585 roh / 0.646 mit Liquor-Regel, s1 0.553 / 0.590 (Abschnitt 3, 5u); Zwei-Startwerte-Ensemble 0.647.
**Entscheidungsregel, vorab:** ein Arm gilt als Gewinn, wenn (i) BEIDE Startwerte gepaart über ihrer jeweiligen Basis liegen und (ii) das Zwei-Startwerte-Ensemble des Arms gepaart gegen das Ensemble der Basis ein Intervall über Null hat (feste Zelle 0.3 / 2 mm3 mit Liquor-Regel, `operating_point` + `ensemble`). Vorab benannte Nebenmaße: Sensitivität für Blutungen unter 10 mm3 und für die Klasse ">5" (dort liegt die Schwäche aus 5ad), Fehlalarme je Fall. Ein Arm, der Sensitivität kauft und F1 verliert (Muster von blob/Zentrum, 5j/5k), gilt als negativ, auch wenn die Nebenmaße steigen. Erwartung, festgehalten: Arm 1 hebt die Sensitivität kleiner Blutungen; ob das F1 folgt, hängt daran, ob die Verwechsler-Beispiele die Fehlalarme in Schach halten -- ich gebe dem 50 %. Arm 2: kleiner Effekt, eher positiv in Kohorte 3, unter 0.03 nicht deutbar.
**Stand 28.09. 19:30 -- gebaut, geprüft, eingereiht:**
- `cmb/synthesis/lesion_stats.py` -> `ergebnisse/analysis/lesion_stats_valdo_0928.json` (139 echte Blutungen: Volumen p10/50/90 = 4.7 / 10 / 45 mm3; Spitzenkontrast 0.094 / 0.201 / 0.275; Mittelkontrast 0.051 / 0.102 / 0.156; z-Ausdehnung 2 / 3 / 6 Schichten; Gewebe Rinde 40, Mark 41, tiefe Kerne 38, infratentoriell 15, Liquor 5).
- `cmb/synthesis/build_synthetic_grid.py` -> `dev/gitter_s`: 57 Fälle, alle Prüfungen bestanden (echtes Label erhalten, Hirnmaske gleich, Komponentenzahl = echt + künstlich), 209 künstliche Blutungen, 319 Verwechsler, Startwert 0, ~10 s je Fall. Erster Entwurf hatte bei dicken Schichten viermal zu große Labels (Blockfüllung auf das gezogene Volumen obendrauf); korrigiert: bei dicken Schichten ist die Grundfläche = Volumen / Schichtdicke, wie beim echten Label. Nachgemessen mit demselben Ring-Verfahren an fünf Probefällen (25 künstliche): Ko1 Volumen 9 / 9 / 15 (echt 7 / 12.5 / 83), Spitze 0.17 / 0.19 / 0.27 (echt 0.09 / 0.17 / 0.27), z 4 / 4 / 4 (echt 4 / 4 / 8); Ko2 5.8 (echt 7.3), Spitze 0.185 (0.228); Ko3 15.8 (16), Spitze 0.176 (0.124). Klein und im echten Kontrastbereich, wie beabsichtigt; Ko3 etwas kontrastreicher als echt. Prüfabbildung `ergebnisse/analysis/synthese_qc/gitter_s_0928.png` (echte und künstliche Blutungen nebeneinander, Verwechsler als kurze dunkle Röhren).
- `turnier.py --synthese` (Sicherung `.vor-synthese-0928`): Wrapper-Test auf 8 Fällen -- echte Daten nach dem Ziehen bitgleich, Testfälle nie geladen, Komponenten je Trainingsfall echt -> künstlich z. B. 0 -> 3, 2 -> 4, 3 -> 9, Stichprobe verschieden, Zwischenspeicher wird beim Faltenwechsel umgebaut.
- `dev/gitter_k` (kubisch): 57 Fälle, Label und Hirnmaske bitgleich zu `gitter_d`, mittlere Bildabweichung im Hirn 0.011, mittlere Läsionsintensität 0.797 -> 0.814 (Beschreibung, kein Entscheider).
- Eingereiht: [203] gitter_k s42, [204] gitter_k s1, [205] gitter_d + Synthese s42, [206] dito s1 -- je ~1.1 h; [203] läuft seit 14:47. Auswertung wie in 5ae festgelegt.
- Arm 3: CSIRO-Sammlung csiro:50304 hat 4073 Dateien / 17.7 GB (NoCMBSubject 313, rCMB_DefiniteSubject 57, sCMB_* 3700 synthetische Varianten, 3 xlsx mit Orten); Lizenz "CSIRO Data Licence": nur nicht-kommerziell, mit Quellenangabe, KEINE Weitergabe -- geholt wird nur rCMB + xlsx nach `~/data/extern/momeni-csiro-cmb/` (LIESMICH.md dort). Der S3-Host `s3.data.csiro.au` ist über den lokalen Resolver nicht auflösbar (Cloudflare-CNAME), Abhilfe `curl --resolve` mit der Adresse aus 1.1.1.1.
- **Arm 3, Stand 28.09. 21:00:** 57 Dateien + 3 Tabellen geholt (`rCMB_locations.json`: 146 Blutungen in 57 SWI, Median 1 je Fall, max 17; Koordinaten (x, y, z) mit Basis 1 -- geprüft: an diesen Voxeln ist das Bild 5.5 SD dunkler als die Umgebung, jede andere Achsenfolge/Basis liegt bei 0). Bilder 176 x 256 x 80, 0.94 x 0.94 x 1.75 mm, LPS, schädelfrei, biaskorrigiert und von den Autoren histogramm-angeglichen ("BFC_50mm_HM"); Dateinamen tragen Zeitpunkte T0-T3 desselben Probanden (z. B. 122_T1 / 122_T2) -- die Fälle sind NICHT unabhängig, für Intervalle müsste je Proband gebündelt werden. Gitter gebaut mit `cmb/transfer/build_momeni_grid.py` (dieselben Funktionen wie VALDO/catalina; Label = Kugel r 1.5 mm um jeden Ort, Ersatz für fehlende Masken): 57 Fälle, 146/146 Läsionen auf dem Gitter. Eingereiht [207]-[209]: Zero-shot des catalina-SWI-Modells (s42, s1) und zur Einordnung des VALDO-T2*-Modells. **Erwartung vorab:** SWI-Modell roh 0.30-0.55 (eigene CV-Zahl 0.626 nur Schwelle; fremder Scanner, Histogramm-Angleich, Punkt-Labels mit Berührungsregel); VALDO-Modell deutlich darunter (auf catalina-SWI 0.305). Bewertet wird primär nach Challenge-Regel (Zentrumsabstand 5 mm, `subject_level_f1`), weil die Referenz Punkte sind; ohne Liquor-Regel (kein SynthSeg). Zweite Stufe nicht anwendbar (kein Anwendungsmodus; nur geschachtelte CV).
**Arm 4 -- Klassifikator auf VALDO vortrainieren, auf catalina feintunen (Nutzeridee 28.09. ~18:00; vorab festgelegt VOR dem Lauf).**
Frage: eigenes U-Net je Kohorte (gemessen richtig, 5h/5n), aber der Klassifikator NICHT gemeinsam auf dem Pool, sondern auf den 278 VALDO-Kandidaten vortrainiert und auf den catalina-Trainingsfalten feingetunt. Warum das trotz gleicher Daten wichtig ist: die heutige Kette braucht die privaten Kandidaten beim Training und ist nicht veröffentlichbar; diese Kette wäre es (U-Net + Klassifikator auf VALDO öffentlich, Klinik trainiert eigenes U-Net und tunt nur den Klassifikator). Klarstellung zur Frage des Nutzers: das gemeinsame 3D-CNN sieht NUR die Ausschnitte (Bild, FRST, Wahrscheinlichkeit; `candidates.py` Z.136), kein Gewebemerkmal -- die Gewebeklasse geht nur in die Merkmalsmodelle ein. Sein SWI-Gewinn kommt daher, dass es die harte Liquor-Regel nicht braucht und das Aussehen aus dem größeren Pool lernt (meine Aussage im Gespräch "lernt mit dem Gewebemerkmal" war falsch).
Umsetzung: `classifier.fit_predict_cnn_transfer` (gleiche Schichten, Ausschnitt, Jitter, Spiegelungen, drei gemittelte Mitglieder wie `fit_predict_cnn`; Vortraining 40 Epochen lr 1e-3 auf der Quelle, Feintuning 20 Epochen lr 3e-4; Vortraining je Startwert einmal, zwischengespeichert -- exakt, weil die Quelle nicht von der Zielfalte abhängt) und `pooled.py --transfer valdo=t2star,swi --fractions 0 0.1 0.25 0.5 1`: je Zielfalte k wird auf einem FALLWEISEN Anteil der catalina-Trainingsfalten feingetunt (geschachtelte Teilmengen, dieselbe Permutation für alle Anteile), Schwelle aus inneren Falten dieser Kandidaten; Anteil 0 = reiner VALDO-Klassifikator mit Schwelle aus VALDOs eigenen inneren Falten (Zero-shot der zweiten Stufe). Sicherungen `.vor-transfer-0928`. Rauchtest mit 1 Epoche auf der CPU vor dem Einreihen.
Läufe: 3 Startwert-Versätze (0, 1, 2), je ein Aufruf mit `--models cnn` (gemeinsames CNN als Bezug im selben Lauf, gepaart je Fall) und den fünf Anteilen; Kandidatensätze S1-S3 (`-weich`). Geschätzt 1.5-2 h GPU je Versatz, hinter [209].
**Erwartung vorab:** (1) Feintuning mit Anteil 1 liegt innerhalb ±0.01 des gemeinsamen CNN (gleiche Daten, andere Reihenfolge; Rauschmaß der zweiten Stufe sd 0.003-0.009 aus 5z); (2) Zero-shot (Anteil 0) liegt darunter, auf SWI deutlich (VALDO sieht keine SWI-Ausschnitte; U-Net-Zero-shot auf SWI war 0.305 gegen 0.626), auf T2* wenig; (3) die Lernkurve erreicht auf T2* bei 25 % das Niveau (T2* ähnelt VALDO), auf SWI erst bei 50 %. **Entscheidung:** ein Anteil gilt als "ausreichend", wenn sein Mittel über die drei Versätze innerhalb 0.01 des gemeinsamen CNN liegt. Bewertet wird die Kurve, nicht ein Einzelwert; ein Gewinn ÜBER dem gemeinsamen CNN wird nicht erwartet und wäre erst mit 10 Versätzen glaubhaft.
**ZWISCHENSTAND 28.09. 21:00 (Nutzerwunsch; Regeln unverändert; `ergebnisse/analysis/5ae_0929/zwischen_*.json`):**
| | Basis s42 | kubisch s42 | Basis s1 | kubisch s1 | Basis Ens. | kubisch Ens. | Synthese s42 |
|---|---|---|---|---|---|---|---|
| feste Zelle 0.3 / 2 mm3 + Liquor-Regel | 0.646 | 0.573 | 0.590 | 0.606 | 0.647 | 0.611 | **0.467** |
| kreuzweise gewählt, gepaart gegen Basis | 0.622 | 0.605, -0.018 [-0.079; +0.043] | 0.564 | 0.607, +0.043 [-0.014; +0.097] | 0.647 | 0.623, -0.024 [-0.086; +0.029] | 0.468, **-0.154 [-0.287; -0.038]** |
**Arm 2 (kubisch) ist damit ENTSCHIEDEN: kein Gewinn.** Regel (i) verletzt (s42 unter der Basis), Regel (ii) verletzt (Ensemble-Intervall um Null, Punktwert negativ). Bemerkenswert nur: das kubische Netz will höhere Schwellen (0.5-0.7 statt 0.3) -- die Kalibrierung verschiebt sich, das F1 nicht. Der zurückgeholte Kontrast (Läsionsintensität +0.017) ist offenbar kein Engpass. Kubisch wird nicht übernommen.
**Arm 1 (Synthese), Startwert 42: deutlich SCHLECHTER, und zwar anders als erwartet.** Erwartet war "mehr Sensitivität, vielleicht mehr Fehlalarme"; gemessen ist WENIGER Sensitivität (0.45 gegen 0.64) UND weniger Precision (0.49 gegen 0.60). Je Kohorte roh: Ko1 -0.184, Ko2 -0.087, Ko3 -0.203; Sensitivität kleiner Blutungen (< 7 mm3) 0.52 -> 0.34, kontrastarmer (unter Median) 0.51 -> 0.30 -- genau die Gruppe, der die Synthese helfen sollte, verliert am meisten. Inneres F1 (echte Wahlfälle) in vier von fünf Falten niedriger; Trainingsverlust höher (0.142 gegen 0.102 in Epoche 60), also keine triviale Aufgabe, sondern eine, die das Netz vom Echten wegzieht. Geprüft: das Synthesegitter wurde geladen (33-36 Trainingsfälle je Falte laut Lauf-Log), Test- und Wahlfalte echt. Regel (i) ist damit schon verletzt, Startwert 1 (läuft, [206]) kann den Arm nicht mehr retten; er liefert nur noch das Rauschmaß.
**Hypothesen, vor der Diagnose notiert:** (H1) Budget: 60 % der Läsionsausschnitte sind jetzt künstlich, die Zahl echter Läsionsausschnitte je Epoche sinkt auf ~40 % -- nach 5h ist das Budget je echter Läsion entscheidend; (H2) die künstlichen Blutungen sind vom Echten unterscheidbar (glatte Gauß-Form auf verrauschtem Hintergrund, blockgefülltes Label auf weichem Bild) und das Netz lernt "künstlich" als eigene, leichte Klasse, während die echten kleinen Blutungen relativ seltener und schwerer werden; (H3) die Verwechsler (helle Röhren, Label 0) heben die Latte für kleine helle Punkte allgemein. Diagnose läuft auf der CPU: Fold-0-Modelle (Basis, Synthese) auf drei Trainingsfälle des Synthesegitters -- Höchstwahrscheinlichkeit je echter Blutung, je künstlicher Blutung, je Verwechsler.
**Diagnose 28.09. 21:35 (`ergebnisse/analysis/5ae_0929/zwischen_syn_probe.txt`; drei Trainingsfälle der Falte 0, je eine Kohorte; Fold-0-Modelle Basis und Synthese; Höchstwahrscheinlichkeit je Komponente):**
| Fall | Modell | echte CMB | künstliche CMB | Verwechsler |
|---|---|---|---|---|
| sub-103 (Ko1, 6 echt / 3 künstlich / 8 Verw.) | Basis | 0.59 0.62 1 1 1 1 | 0.67 1 1 | alle 0.00 |
| | Synthese | 0.22 0.62 1 1 1 1 | 0.85 1 1 | alle 0.00 |
| sub-218 (Ko2, 26 / 5 / 3) | Basis | 0 0 .25 .33 .41 .50 .50 .62 .75 .83 .90 .96 .99 1 ... | 0 0 0 .13 1 | alle 0.00 |
| | Synthese | 0 0 0 .25 .25 .33 .34 .39 .43 .50 .50 .50 .75 .86 .88 1 ... | .25 .51 .84 .87 1 | alle 0.00 |
| sub-305 (Ko3, 2 / 3 / 8) | Basis | .75 1 | .50 1 1 | 0 ... 0.50 |
| | Synthese | 1 1 | .50 1 1 | 0 ... 0.23 |
**Gelesen:** (H3 VERWORFEN) die Verwechsler bekommen von BEIDEN Modellen 0.00 -- auch von der Basis, die nie einen gesehen hat; helle Gauß-Röhren sind keine harten Negativen, das Netz hat sie nie mit Blutungen verwechselt. (H2 nur für Kohorte 2) die Basis findet in Ko1/Ko3 fast alle künstlichen Blutungen (0.5-1.0), ohne je eine gesehen zu haben -- sie sind dort realistisch genug; in Ko2 (0.8 mm) sind 3 von 5 für die Basis unsichtbar (zu klein, zu glatt, ohne den scharfen Rand echter Blutungen bei hoher Auflösung). (H1 GESTÜTZT) auf denselben TRAININGSfällen gibt das Synthesemodell den echten Blutungen NIEDRIGERE Wahrscheinlichkeiten als die Basis (sub-218: 0.00/0.00/0.00/0.25/0.25/0.33 gegen 0.00/0.00/0.25/0.33/0.41/0.50) -- es hat die echten Blutungen schlechter gelernt, weil 60 % seiner Läsionsausschnitte künstlich waren und das Budget je ECHTER Läsion (5h) gesunken ist.
**Arm 1, erste Fassung: NEGATIV, geschlossen.** Regel (i) verletzt; Startwert 1 ([206]) läuft aus und liefert das Rauschmaß.
**Zweite Fassung (5ae-1b), vorab festgelegt 28.09. 21:40:** (1) Budget der echten Läsionen UNVERÄNDERT: die Stichprobe wird wie in der Basis gezogen (60 % echte Läsion / 25 % Hirn / 15 % beliebig); danach wird die Hälfte der Ausschnitte OHNE Läsion durch Ausschnitte um künstliche Blutungen ersetzt (`--synthese-anteil 0.5` -> 20 % aller Ausschnitte künstlich, echte Läsionsausschnitte exakt wie bisher); (2) keine Verwechsler; (3) flacheres Profil mit schärferem Rand (Super-Gauß, `--profile flat`) gegen die Unsichtbarkeit in Ko2; (4) sonst wie 5ae (N ~ U{2..6}, Größen/Kontrast/Ort aus der echten Verteilung). Gitter `dev/gitter_s2`. Erwartung: Sensitivität kleiner Blutungen steigt leicht, F1 innerhalb des Rauschens (±0.03); Gewinn zählt weiter nur nach Regel (i)+(ii). Eingereiht hinter Arm 4.
**ERGEBNIS ARME 1-3 (automatische Auswertung 28.09. 22:17, `ergebnisse/analysis/5ae_0929/`; gelesen 22:45 gegen die Regeln oben):**
| feste Zelle 0.3 / 2 mm3 + Liquor-Regel; gepaart = kreuzweise gewählt | s42 | s1 | Zwei-Startwerte-Ensemble |
|---|---|---|---|
| Basis | 0.646 | 0.590 | 0.647 (P 0.65, S 0.64, FP 0.82) |
| Arm 2 kubisch | 0.573, gepaart -0.018 [-0.079; +0.043] | 0.606, +0.043 [-0.014; +0.097] | 0.611, **-0.024 [-0.086; +0.029]** (P 0.57, S 0.68, FP 1.25) |
| Arm 1 Synthese, erste Fassung | 0.467, -0.154 [-0.287; -0.038] | 0.374, -0.206 [-0.274; -0.128] | 0.473, **-0.130 [-0.212; -0.056]** (P 0.55, S 0.49) |
Nebenmaße (Ensembles, `missed_ens2.txt`): Sensitivität klein (< 7.2 mm3) Basis 0.50 / kubisch **0.68** / Synthese 0.36; kontrastarm (unter Median) 0.46 / **0.62** / 0.33; groß 0.74 / 0.87 / 0.72. Fehlalarme je Fall je Kohorte (roh): Basis 1.25 / 1.59 / 0.95, kubisch 2.50 / 3.15 / 2.91, Synthese 2.25 / 1.85 / 3.86.
**Arm 1 (erste Fassung): NEGATIV, gesichert, beide Startwerte und Ensemble.** Diagnose und zweite Fassung oben (5ae-1b, [213]-[214]).
**Arm 2 (kubisch): KEIN GEWINN nach Regel (i) (s42 unter der Basis) und (ii) (Ensemble-Intervall um Null, Punktwert -0.024).** Bemerkenswert, aber vorab als "negativ" eingestuftes Muster: die kubische Interpolation macht das Netz EMPFINDLICHER (kleine Blutungen 0.50 -> 0.68, kontrastarme 0.46 -> 0.62, gesamt S 0.64 -> 0.68) und bezahlt mit Fehlalarmen (0.82 -> 1.25 je Fall, in Kohorte 2 verdoppelt) -- exakt das Muster von blob/Zentrum (5j/5k): Sensibilität kaufen, F1 verlieren. Das Netz will höhere Schwellen (0.5-0.7); an seiner eigenen kreuzweise gewählten Zelle ist es trotzdem nicht besser (0.623 gegen 0.647). Nicht übernommen. Kubisch bleibt als Option, falls jemals ein Klassifikator die zusätzlichen Kandidaten trennen kann (Kandidaten-Decke 5w) -- heute kann er es nicht (5y).
**Arm 3 (Momeni/CSIRO, 57 SWI, 146 Punkt-Referenzen, Kugel-Label 1.5 mm, ohne Liquor-Regel, `momeni_*.txt`):**
| Modell (Zero-shot) | gepoolt F1 | P | S | FP je Fall | Median F1 je Proband (Challenge-Regel, 5 mm) |
|---|---|---|---|---|---|
| catalina-SWI s42 | **0.491** | 0.36 | 0.79 | 3.63 | 0.50 |
| catalina-SWI s1 | 0.407 (gepaart -0.084 [-0.108; -0.062]) | 0.28 | 0.77 | 5.19 | 0.40 |
| catalina-SWI Ensemble s42+s1 | 0.447 | 0.32 | 0.77 | 4.26 | 0.50 |
| **VALDO-T2* s42** | **0.525** (gepaart gegen SWI s42 +0.034 [-0.011; +0.077]) | 0.39 | 0.81 | 3.30 | 0.50 |
Erwartung geprüft: SWI-Modell 0.30-0.55 -- eingetroffen (0.49). "VALDO-Modell deutlich darunter" -- **FALSCH**: das VALDO-T2*-Modell ist auf dieser fremden SWI-Kohorte gleichauf bis besser als das eigene SWI-Modell (auf catalina-SWI war es 0.305 gegen 0.626). Gelesen: (1) beide Modelle finden ~80 % der markierten Blutungen ohne jede Anpassung -- die erste UNABHÄNGIGE öffentliche Zahl der Linie, und sie liegt bei Sensitivität auf VALDO-Niveau; (2) der Abstand steckt in der Precision (0.36-0.39, 3.3-3.6 Fehlalarme je Fall), und dafür gibt es drei Verdächtige, die die Zahl nach oben korrigieren würden: keine Liquor-Regel (auf VALDO +0.06), Punkt-Referenz nur der "definite" Blutungen (mögliche/unsichere sind nicht markiert und zählen als Fehlalarm), und 1.5-mm-Kugeln statt Masken (Berührungsregel strenger als bei echten Masken); (3) das SWI-Ensemble ist SCHLECHTER als sein bester Startwert (0.447 gegen 0.491), weil s1 hier deutlich abfällt -- die Zwei-Startwerte-Regel (5p: "nie den besten Startwert ausliefern") bleibt trotzdem, denn welcher Startwert auf einer fremden Kohorte vorn liegt, weiß man vorher nicht (auf catalina-T2* war es s1); (4) das VALDO-Modell überträgt auf 1.75-mm-SWI besser als auf die 2-mm-catalina-SWI -- die Momeni-Bilder sind histogramm-angeglichen und schädelfrei wie VALDO, catalina-SWI hat den schwächsten Kontrast (5n); der Kohortenunterschied ist also nicht "SWI gegen T2*", sondern Bildstatistik. Grenzen: Zeitpunkte desselben Probanden zählen als Fälle (Intervalle zu eng); keine Liquor-Regel (SynthSeg auf 57 Fällen wäre ~2 CPU-Stunden mit `--parallel 2`); Referenz = Punkte.
**Folgerung für Arm 3:** als externe Validierung ins Paper (Sensitivität ~0.8, F1 ~0.5 ohne Anpassung), mit den drei Vorbehalten; SynthSeg + Liquor-Regel nachziehen, sobald die GPU-Schlange leer ist (CPU-Last neben dem Training vermeiden).
**Nachgezogen 29.09. 00:10 -- SynthSeg --robust auf den 57 Momeni-Fällen (`cmb/transfer/momeni_synthseg.py`, 22 s je Fall, CPU; Karten auf dem Gitter unter `<extern>/synthseg/`) und Liquor-Regel (`momeni_op_csf.json`, `momeni_subject_csf.json`):**
| Modell (Zero-shot, feste Zelle + Liquor-Regel) | F1 gepoolt | P | S | FP je Fall | Median F1 je Proband | ohne Regel (oben) |
|---|---|---|---|---|---|---|
| catalina-SWI s42 | **0.555** | 0.49 | 0.64 | 1.68 | 0.667 | 0.491 |
| catalina-SWI s1 | 0.509 | 0.42 | 0.64 | 2.26 | 0.533 | 0.407 |
| catalina-SWI Ensemble | 0.522 (kreuzweise 0.570) | 0.44 | 0.64 | 2.07 | 0.571 | 0.447 |
| **VALDO-T2* s42** | **0.613** (gepaart gegen SWI s42 +0.028 [-0.031; +0.090]) | 0.58 | 0.65 | 1.21 | **0.667** | 0.525 |
Die Liquor-Regel hebt hier alle vier Modelle um 0.06-0.10, wie auf VALDO -- aber anders als dort NICHT umsonst: die Sensitivität fällt von ~0.80 auf ~0.64, die Regel löscht also rund ein Fünftel der markierten Blutungen (auf VALDO 0 von 139). Vermutlich liegen bei Momeni mehr markierte Blutungen an Sulci (Punkt-Referenz, 1.75-mm-Schichten, engere Schädelfrei-Maske: 36 % der Bildvoxel tragen kein SynthSeg-Label). Für Momeni wäre die WEICHE Fassung (gemeinsames CNN mit Gewebemerkmal, 5e) der richtige Weg, sie ist ohne Anwendungsmodus aber nicht verfügbar (Arm 4 baut genau den).
**Bilanz Arm 3, so ins Paper:** externe Validierung auf einer fremden öffentlichen SWI-Kohorte ohne jede Anpassung -- Median-F1 je Proband 0.667, gepoolt 0.55-0.61 mit Liquor-Regel bzw. 0.49-0.53 ohne (Sensitivität 0.8), das VALDO-Modell gleichauf oder besser als das SWI-Modell. Vorbehalte: Punkt-Referenz nur der sicheren Blutungen, Kugel-Label, Zeitpunkte desselben Probanden, Liquor-Regel kostet hier Sensitivität.
**ERGEBNIS ARM 4 (29.09. 06:50; `mb-arena/stage2/transfer_valdo_{0,1,2}.json`, drei Startwert-Versätze, Mittel; Differenz zum gemeinsamen CNN desselben Laufs):**
| Klassifikator | catalina-T2* F1 (Differenz) | catalina-SWI F1 (Differenz) |
|---|---|---|
| nur Schwelle | 0.654 (-0.016) | 0.626 (-0.046) |
| Schwelle + Liquor-Regel | 0.670 (0.000) | 0.614 (-0.058) |
| gemeinsames CNN (Pool aller Kohorten) | **0.670** | **0.672** |
| VALDO-Klassifikator, Zero-shot (Anteil 0) | 0.648 (-0.022; P 0.75, S 0.57) | **0.671 (0.000)** |
| feingetunt mit 10 % der Trainingsfälle | 0.606 (-0.064; ein Versatz 0.65, zwei 0.58) | 0.620 (-0.051) |
| 25 % | 0.644 (-0.025) | 0.651 (-0.021) |
| 50 % | 0.666 (-0.004) | 0.662 (-0.009) |
| 100 % | 0.670 (0.000) | 0.660 (-0.012) |
**Gelesen gegen die Erwartungen:** (1) "Anteil 1 innerhalb ±0.01 des gemeinsamen CNN" -- eingetroffen (T2* 0.000, SWI -0.012 knapp daneben). (2) "Zero-shot auf SWI deutlich darunter" -- **FALSCH**: der reine VALDO-Klassifikator ist auf SWI GLEICHAUF mit dem gemeinsamen CNN (0.671 gegen 0.672, in allen drei Versätzen innerhalb 0.008), auf T2* nur 0.022 darunter -- und dort mit HÖHERER Precision (0.75) bei niedrigerer Sensitivität: die VALDO-Schwelle ist für T2* zu streng, der Klassifikator selbst überträgt. (3) "25 % reichen auf T2*, 50 % auf SWI" -- halb: T2* braucht 50 %; auf SWI bringt Feintuning gar nichts, und mit 10-25 % der Fälle SCHADET es (Überanpassung an wenige Fälle plus instabile Schwelle aus wenigen inneren Kandidaten; ein Versatz fällt auf 0.562). Nebenbefund: der SWI-Gewinn des gemeinsamen CNN gegen Schwelle + Liquor-Regel ist mit drei weiteren Versätzen erneut da (+0.058), der Startwert-Abstand des CNN liegt bei 0.003-0.013.
**Folgerung, die zählt:** die VERÖFFENTLICHBARE Kette (U-Net + Klassifikator nur auf VALDO trainiert, ohne jede private Kandidate) ist auf beiden catalina-Kohorten so gut wie die heutige private Pool-Kette (SWI gleich, T2* -0.022 mit zu strenger Schwelle). Das ist die Kette, die ins öffentliche Repo gehört; Feintuning auf lokalen Kandidaten lohnt erst ab etwa der Hälfte der Kohorte und bringt dann höchstens 0.01-0.02. Für Momeni (Arm 3) folgt daraus: der VALDO-Klassifikator kann dort als weiche Gewebe-Alternative zur harten Liquor-Regel angewendet werden -- ein Anwendungsmodus (Klassifikator auf allen VALDO-Kandidaten trainieren, auf fremde Kandidaten anwenden) ist jetzt mit `fit_predict_cnn_transfer` (Anteil 0) trivial zu bauen. Nächster Schritt, wenn gewünscht.
**ERGEBNIS ARM 1, ZWEITE FASSUNG (5ae-1b; 29.09. 06:50; `synv2_op_*.json`, `synv2_missed_ens2.json`):** feste Zelle + Liquor-Regel s42 0.590 gegen 0.646 (kreuzweise gepaart -0.024 [-0.088; +0.044]), s1 0.532 gegen 0.590 (-0.041 [-0.105; +0.021]), Ensemble 0.605 gegen 0.647 (**-0.072 [-0.129; -0.018]**, P 0.55 S 0.60 FP 1.21 gegen P 0.65 S 0.64 FP 0.82). Nebenmaße wie erwartet: Sensitivität klein 0.50 -> 0.57, kontrastarm 0.46 -> 0.51, groß 0.74 -> 0.83 -- aber bezahlt mit Fehlalarmen. **Regel (i) und (ii) verletzt: negativ.** Die zweite Fassung hat die Diagnose bestätigt (Budget erhalten -> kein Einbruch mehr wie in Fassung 1, -0.13 -> -0.04 am Ensemble roh) und zeigt dann dasselbe Muster wie jeder Trainingshebel dieser Linie (5j/5k, kubisch 5ae): mehr Empfindlichkeit, weniger F1. **Arm 1 ist geschlossen.** Bilanz: künstliche Blutungen in der ersten Stufe geben dem Netz keine Information, die Blutung von Verwechsler trennt; sie senken nur die Latte. CenSynCMBs Gewinn kam zusammen mit T1 + T2 + T2* und gelabelten Verwechslern, die bei uns nichts lehrten -- unser Befund widerspricht ihrem nicht, er zeigt die Bedingung.
**Arm 5 -- Gefäße als Fehlalarme: dunkle Laufstrecke entlang z (Nutzerbeobachtung 29.09. im Schichtbetrachter: "viele Fehlalarme sind Gefäße, die über mehrere Schichten laufen; Blutungen sind in z nicht so lang"; vorab festgelegt 29.09. ~08:50).**
Warum das Neues sein könnte: FRST ist 2D je Schicht und kann Gefäßquerschnitt und Blutung nicht trennen; das Merkmal "Ausdehnung durch die Schichten" der Merkmalsmodelle maß nur die Komponente NACH der Schwelle (ein Gefäß zerfällt bei 0.15 in Stücke); der zweiskalige Klassifikator senkte auf T2* die Fehlalarme (3.0 -> 2.7), also gibt es dort etwas zu holen; Übermalung (Original) löschte Blutungen mit und ist kein Weg.
Messung (`cmb/analysis/vessel_run_length.py`, CPU): Kandidaten = Komponenten der out-of-fold-Karte bei 0.15 / >= 1 mm3 (wie die zweite Stufe), je Kandidat: (a) z-Ausdehnung der Komponente selbst, (b) **dunkle Laufstrecke im BILD**: von der Kandidatenmitte aus Schicht für Schicht dem lokalen Helligkeitsmaximum der invertierten Skala folgen (Fenster ±1 mm in der Schicht für schräge Gefäße), solange der Kontrast zum Ring (1-2 mm) mindestens die Hälfte des Kontrasts an der Mitte hält; Länge in mm nach oben plus unten plus 1. Echte Blutungen haben im Gitter median 3 Schichten z-Ausdehnung, p90 6 (Ko1 4-8 durch die Blockfüllung). Ausgewertet auf VALDO (57), catalina-T2* (75), catalina-SWI (61), Momeni (57): Verteilung der Laufstrecke für echte gegen falsche Kandidaten (Perzentile, AUC), dann die Regel "Laufstrecke > L mm -> verwerfen" mit L aus {6, 8, 10, 12, 15, 20} je Falte auf den ANDEREN Falten gewählt, gepaart gegen die feste Zelle und gegen feste Zelle + Liquor-Regel.
**Erwartung vorab:** die Laufstrecke trennt echte von falschen Kandidaten mit AUC 0.65-0.75 (nicht alle Fehlalarme sind Gefäße; 5i: viele liegen in Sulci/Liquor, die die Regel schon trifft); als Regel +0.01 bis +0.03 F1 auf T2*/SWI (5-mm- bzw. 2-mm-Schichten: dort laufen Gefäße durch wenige Schichten, der Effekt ist begrenzt) und +0.00 bis +0.02 auf VALDO; TP-Verlust unter 3 %. Gewinn zählt, wenn er gepaart über Null liegt UND weniger als 3 % der Referenzblutungen verwirft. Fällt die AUC unter 0.6, ist die Beobachtung qualitativ richtig, aber nicht der Hauptanteil der Fehlalarme, und Schritt 3 (länglicher Klassifikator-Ausschnitt) wird trotzdem gemessen, weil der Klassifikator mehr sieht als eine Laufstrecke.
**Ergebnis Schritt 1+2 (29.09. ~09:10; `ergebnisse/analysis/5ae_0929/vessel_*.json`; Kandidaten der festen Zelle; Regel je Falte auf den anderen Falten gewählt):**
| Kohorte (erste Stufe) | echte / falsche Kandidaten | Laufstrecke echt p50 / p90 | falsch p50 / p90 | AUC (falsch länger) | Regel auf fester Zelle | Regel auf Zelle + Liquor | TP-Verlust |
|---|---|---|---|---|---|---|---|
| VALDO (Ensemble) | 95 / 95 | 5 / 15 mm | 6 / 21 mm | 0.55 | -0.021 [-0.056; +0.014] | -0.009 [-0.031; +0.015] | 11 % / 6 % |
| catalina-T2* (eigenes) | 562 / 309 | 11 / 23 mm | 11 / 23 mm | **0.47** (echte eher länger) | **-0.047 gesichert** | -0.054 gesichert | 13 % |
| catalina-SWI (eigenes) | 435 / 290 | 7 / 16 mm | 7 / 30 mm | 0.55 | -0.004 n. s. | -0.015 n. s. | 7 % |
| Momeni (VALDO-U-Net) | 123 / 224 | 5 / 8 mm | 9 / 28 mm | **0.72** | **+0.108 [+0.078; +0.142]** (0.499 -> 0.607) | **+0.044 [+0.022; +0.071]** (0.601 -> 0.645), L = 10 mm | 6 % / **2 %** |
z-Ausdehnung der Komponente selbst: AUC 0.27-0.49 überall, echte sind dort LÄNGER (Blockfüllung dicker Schichten) -- als Merkmal unbrauchbar, wie vermutet.
**Gelesen:** die Beobachtung ist auf Momeni quantitativ richtig und trägt: falsche Kandidaten laufen dort median 9 mm, p90 28 mm, echte 5 / 8 mm; die Regel "länger als 10 mm verwerfen" hebt das F1 auch NACH der Liquor-Regel gesichert um 0.044 bei 2 % TP-Verlust -- nach der vorab festgelegten Regel ein Gewinn, und 0.645 ist die beste Momeni-Zahl (über dem VALDO-Klassifikator mit 0.607). Auf VALDO und catalina trennt die Laufstrecke dagegen nicht (AUC 0.47-0.55), und auf T2* sind die ECHTEN Kandidaten eher länger: bei 5-mm-Schichten und ausgeprägtem Blooming reicht jede Blutung über 2-3 Schichten = 10-15 mm, und die Verfolgung läuft durch. Auch auf VALDO stammen 10 der 23 langen echten Kandidaten aus Kohorte 1 (4 mm). Von den 27 langen falschen VALDO-Kandidaten liegen 9 im Liquor -- die Liquor-Regel trifft einen Teil derselben Gefäße schon.
**Entscheidung nach Vorabregel:** Regel NICHT in die allgemeine Kette (auf drei von vier Kohorten neutral bis gesichert schlechter). Für dünnschichtige, kontrastreiche SWI wie Momeni ist sie wirksam; sie wird als optionaler Schalter geführt (`vessel_run_length.py`, L aus den inneren Falten der Zielkohorte). Die Erwartung "AUC 0.65-0.75 überall" war zu optimistisch; sie traf nur auf Momeni zu. Der einfache Verfolger ist nicht die allgemeine Lösung -- **Schritt 3 (Klassifikator mit langem z-Ausschnitt) wird gemessen**, weil er das Muster lernen kann, das die Handregel bei dicken Schichten verfehlt.
**Schritt 3, vorab festgelegt (29.09. ~09:20):** Kandidaten mit `--half 20 20 20` (Ausschnitt 20 x 20 x 40 mm, Ordner `-weich-z40`), Klassifikator `cnn-z32` = dasselbe CNN mit Ausschnitt 32 mm entlang z und 16 mm in der Schicht (`fit_predict_cnn(crop=(32, 32, 32))`; Netz ist voll faltend, sonst unverändert), gepaart gegen `cnn` (16-mm-Würfel) auf DENSELBEN Kandidaten im selben Lauf, drei Startwert-Versätze. Erwartung: T2* +0.01 bis +0.02 (dort half auch der zweiskalige Kontext), SWI ±0.01, VALDO ±0.01; Fehlalarme je Fall sinken, Sensitivität bleibt. Gewinn gilt, wenn das Mittel über drei Versätze über +0.01 liegt UND VALDO nicht unter -0.01 fällt (Regel aus 5r). Danach, falls positiv, `apply.py` mit demselben Ausschnitt auf Momeni.
**ERGEBNIS Schritt 3 (29.09. 12:55; `mb-arena/stage2/z32_{0,1,2}.json`, drei Versätze, `cnn-z32` gegen `cnn` auf denselben z40-Kandidaten):**
| Kohorte | CNN 16-mm-Würfel (Mittel) | CNN 32 mm z-Kontext (Mittel) | Differenz je Versatz | Mittel | P / S / FP je Fall |
|---|---|---|---|---|---|
| VALDO | 0.630 | 0.616 | +0.001 / -0.025 / -0.018 | **-0.014** | 0.65 -> 0.64 / 0.61 -> 0.59 / 0.79 -> 0.81 |
| catalina-T2* | 0.669 | 0.681 | +0.007 / +0.011 / +0.017 | **+0.012** | 0.70 -> 0.71 / 0.64 -> 0.66 / 3.08 -> 3.03 |
| catalina-SWI | 0.672 | 0.669 | -0.001 / +0.004 / -0.011 | -0.003 | 0.73 -> 0.71 / 0.62 -> 0.63 / 2.41 -> 2.74 |
**Gelesen gegen die Regel:** T2* erfüllt "> +0.01 im Mittel" (alle drei Versätze positiv; die Startwert-Streuung der zweiten Stufe liegt bei 0.003, der Effekt ist also real), aber VALDO fällt mit -0.014 unter die Grenze von -0.01 -> **nicht in die allgemeine Kette übernommen.** Dasselbe Muster wie beim zweiskaligen CNN (5r: T2* +0.016/+0.019, VALDO -0.030): auf 5-mm-T2* hilft dem Klassifikator mehr Kontext entlang z ein wenig (Gefäßfortsetzung, Blooming über Schichten), auf SWI nichts, und auf VALDO mit nur 278 Kandidaten kostet die größere Eingabe. Für eine reine T2*-Auslieferung ist `cnn-z32` die beste gemessene zweite Stufe (0.681 gegen 0.669 -- und das zweiskalige CNN lag mit 0.687-0.691 noch etwas höher). Arm 5 damit abgeschlossen: die Gefäßfrage ist auf dicken Schichten weder mit Regeln noch mit mehr z-Kontext wesentlich zu bewegen; der Rest ist Physik der Aufnahme.
**Schritt 2b -- Frangi-Gefäßigkeit und Hesse-Kugeligkeit als Nachbearbeitung (Nutzeridee 29.09. ~09:30; `vessel_run_length.py` erweitert, `ergebnisse/analysis/5ae_0929/frangi_*.json`; je Kandidat auf einem lokalen, isotrop 0.5 mm umgerechneten Ausschnitt, Skalen 0.5-1.5 mm, Maximum im 1 mm erweiterten Kandidaten):**
| Kohorte | Frangi-AUC (falsch gefäßiger) | Kugeligkeit-AUC (echt kugeliger) | Regel Gefäßigkeit (geschachtelt) | Regel Kugeligkeit | Regel Kugeligkeit + Laufstrecke |
|---|---|---|---|---|---|
| VALDO | **0.42** (echte sind gefäßiger!) | 0.62 | aus gewählt, ±0 | -0.010 | -0.010 |
| catalina-T2* | **0.45** | 0.58 | aus, ±0 | aus, ±0 | aus, ±0 |
| catalina-SWI | 0.62 | 0.48 | aus, ±0 | aus, ±0 | aus, ±0 |
| Momeni | **0.25** | **0.82** | aus, ±0 | aus, ±0 | = Laufstrecke allein, +0.044 |
**Gelesen:** Frangi kehrt sich um -- die ECHTEN Blutungen sind nach Frangi gefäßiger als die Fehlalarme (Momeni AUC 0.25, VALDO 0.42, T2* 0.45). Der Grund liegt im Gitter: eine Blutung mit Blooming auf 2-5-mm-Schichten wird nach der Umrechnung auf 1 mm zu einem kurzen Zylinder entlang z, und genau darauf springt ein Röhrendetektor an; die feinen Skalen (0.5-1.5 mm) sehen zudem den Rand einer 3-mm-Blutung als Röhre. Die Kugeligkeit trennt auf Momeni gut (AUC 0.82), aber als harte Regel nach der Liquor-Regel bringt sie nirgends etwas (die verbleibenden Fehlalarme sind dort ähnlich kugelig wie die echten); auf VALDO kostet sie. **Entscheidung: Frangi/Kugeligkeit nicht als Regel.** Als Merkmal für den Klassifikator wäre die Kugeligkeit auf dünnschichtigen Daten denkbar, aber das CNN sieht die Form schon selbst -- Schritt 3 misst genau das mit mehr z-Kontext. Werkzeug-Lehre: jeder handgemachte Formfilter muss auf DIESEM Gitter (anisotrop umgerechnet, Blooming) gemessen werden; die Intuition vom isotropen 7-T-Bild überträgt sich nicht.
**Schritt 2c -- Randregel nach Sundaresan 2023 (Kandidaten < 5 mm vom Hirnmaskenrand verwerfen; gemessen 29.09. mittags, `border_*.json`, Abstand per EDT auf dem Gitter, d aus {2, 3, 4, 5, 7 mm} je Falte geschachtelt):** AUC "falsch näher am Rand" 0.49 / 0.49 / 0.51 / 0.48 (VALDO / T2* / SWI / Momeni) -- falsche und echte Kandidaten sitzen gleich weit vom Rand (Median 28 / 17 / 13 / 27 mm). Regel nach Liquor-Regel: +0.002 / -0.002 / 0.000 / 0.000, überall ohne Bedeutung. Grund: bei uns ist der Sulcus-Anteil schon durch die Liquor-Regel (SynthSeg) abgedeckt, und die Hirnmasken (VALDO-Kopfmaske, HD-BET) laufen anders als OXVASCs; die verbleibenden Fehlalarme liegen tief im Parenchym. **Nicht übernommen.** Damit sind von Sundaresans drei Nachbearbeitungsregeln alle drei bei uns geprüft: Volumen (haben wir), Elliptizität (kippt, Frangi-Befund), Rand (wirkungslos). Die Precision 0.84 auf OXVASC entsteht dort aus der Kombination mit Übermalung und Destillation auf 5-mm-Daten in nativer Auflösung -- ein Setting, das wir nur mit dem feingetunten Original selbst nachstellen (Arm 6, läuft).
**Schritt 2d -- Objektgeometrie der Kandidatenkomponente (Nutzerfrage 29.09. nachmittags nach leichter Nachbearbeitung; Literatur `recherche/leichte-nachbearbeitung-0929.md`: Achsverhältnis > 3 = Gefäß in der geometrischen Kaskade 2026, Objektmerkmale bei van den Heuvel 2016; gemessen `geom_*.json`):** je Kandidat PCA-Achsverhältnis (längste/kürzeste Hauptachse in mm), In-plane-Verhältnis je Schicht, Schwerpunktverschiebung zwischen Nachbarschichten, Zahl belegter Quellschicht-Blöcke; Regeln geschachtelt nach der Liquor-Regel.
| Kohorte | AUC Achsverhältnis (falsch länglicher) | AUC in-plane | AUC Verschiebung | Regel Achsverhältnis | Regel in-plane | Regel Verschiebung |
|---|---|---|---|---|---|---|
| VALDO | 0.54 | 0.59 | 0.47 | -0.003 | -0.001 | -0.005 |
| catalina-T2* | 0.56 | 0.61 | 0.53 | +0.000 | +0.001 | 0.000 |
| catalina-SWI | 0.49 | 0.62 | 0.54 | +0.002 | -0.001 | +0.001 |
| Momeni | 0.67 | 0.57 | 0.67 | -0.002 | -0.007 | 0.000 |
Zahl belegter Blöcke: AUC 0.34-0.51, echte belegen MEHR (Blockfüllung), wie schon die z-Ausdehnung. **Nichts davon trägt** -- alle Regeln ±0.005, auch auf Momeni, wo die Laufstrecke im Bild +0.044 brachte. Erklärung: die Komponente ist die des U-Nets bei Schwelle 0.15, nicht die Struktur im Bild; ein Gefäß, das das Netz nur punktweise markiert, ist als Komponente so kompakt wie eine Blutung (Median-Achsverhältnis echt 1.8-2.7 gegen falsch 1.9-2.9). Damit sind alle "leichten" Formhebel aus der Literatur bei uns gemessen: Laufstrecke (nur Momeni), Frangi, Kugeligkeit, Rand, Achsverhältnis, In-plane-Form, Verschiebung, Blockzahl. **Schritt 2 ist geschlossen; für Gefäß-Fehlalarme gibt es auf diesem Gitter keine Regel, nur das gelernte Netz (Liquor-Regel und Klassifikator).**

**Bilanz der Forschungslinie 5ae (vier Arme, ~13 GPU-Stunden):** kein Arm hebt das VALDO-F1; zwei Erwartungen fielen falsch aus, beide in dieselbe Richtung: **die VALDO-Modelle übertragen besser als gedacht** (T2*-U-Net auf Momeni-SWI 0.613 gegen SWI-U-Net 0.555; VALDO-Klassifikator zero-shot auf SWI gleichauf mit dem Pool). Das ist das Ergebnis, das ins Paper gehört: eine rein öffentliche, veröffentlichbare Kette, extern validiert.
**Anwendungsmodus gebaut und auf Momeni angewandt (29.09. 07:40, Nutzerfreigabe; `cmb/stage2/apply.py`, `ergebnisse/analysis/5ae_0929/momeni_apply_valdo_classifier.json`):** Klassifikator (dasselbe CNN, drei Startwerte x drei Mitglieder) auf ALLEN 278 VALDO-Kandidaten trainiert, Schwelle 0.50 aus VALDOs eigenen inneren Falten (out-of-fold F1 dort 0.620), nichts von Momeni für irgendeine Wahl benutzt. Momeni-Kandidaten mit `cmb.stage2.candidates --keep-csf` aus beiden Zero-shot-Läufen (534 bzw. 529 Kandidaten, Obergrenze Sensitivität 0.88 / 0.85).
| erste Stufe auf Momeni | feste Zelle (0.3 / 2 mm3) | + harte Liquor-Regel | + VALDO-Klassifikator | gepaart Klassifikator gegen Regel |
|---|---|---|---|---|
| VALDO-T2*-U-Net s42 | 0.499 (P 0.35, S 0.84) | 0.610 (P 0.55, S 0.69) | 0.607 (P 0.54, S 0.70) | -0.003 [-0.056; +0.050] |
| catalina-SWI-U-Net s42 | 0.455 (P 0.32, S 0.80) | 0.524 (P 0.43, S 0.67) | **0.630 (P 0.59, S 0.67)** | **+0.106 [+0.044; +0.159]** |
**Gelesen:** auf den Kandidaten des VALDO-U-Nets ist der gelernte Klassifikator gleichauf mit der harten Regel (wie auf VALDO selbst, 5e); auf den Kandidaten des SWI-U-Nets schlägt er die harte Regel gesichert um 0.106 und liefert mit 0.630 die beste Momeni-Zahl der Linie -- ohne dass je ein Momeni- oder catalina-Kandidat ins Training ging. Die Sensitivität kostet er wie die Regel (0.80 -> 0.67; die verlorenen Blutungen sind vermutlich dieselben sulcusnahen Punkte), aber bei deutlich besserer Precision. Damit ist die rein öffentliche Kette komplett und extern validiert: VALDO-U-Net -> VALDO-Klassifikator, F1 0.61 / Median je Proband 0.667 auf einer fremden SWI-Kohorte. Für die Auslieferung ist `apply.py` der Modus; `accepted` im JSON hält die angenommenen Kandidaten je Fall für eine Maskenausgabe bereit.

**Nutzerpräferenz 29.09. 10:50:** "das Modell mit der höchsten Sensitivität gefällt mir am besten" -- für die Auslieferung zählt Sensitivität vor Precision (Vorsortierung für den Befunder). Das ist heute das kubische Ensemble mit Liquor-Regel (Sensitivität 0.68 gegen 0.64, kleine Blutungen 0.68 gegen 0.50, Fehlalarme 1.25 statt 0.82 je Fall; F1 0.611 gegen 0.647) bzw. auf catalina-SWI blob (kleine 0.44 gegen 0.37). Folge: neben der F1-Kette eine "empfindliche Kette" führen und beide berichten. Nutzerauftrag gleichzeitig: unsere Kette gegen das FEINGETUNTE Original (MicrobleedNet, vortrainierte Gewichte + fine_tune) auf VALDO und catalina messen (Arm 6, siehe unten) und die Kernexperimente als wiederholbares Python-Skript vorbereiten.

**Arm 6 -- feingetuntes Original als faire Messlatte (Nutzerauftrag 29.09. 10:55).** `cmb/baselines/microbleednet_finetune.py`: das Original (vortrainierte Gewichte, eigene Vorverarbeitung `data_preparation.preprocess_subject`, `fine_tune` mit allen Schichten, 60 Ep., Frühstopp 10, `evaluate`) auf UNSEREN Falten; Auswertung nativ (mb_metrik-Logik) und auf unserem Gitter (Vorhersage nächster-Nachbar umgerechnet, `cmb.transfer.evaluate`). Vorhanden aus der Arena (09/2026, Vorverarbeitung preproc2, alle Schichten): catalina-SWI -- auf unserem Gitter mit unserer Metrik **0.406 (P 0.34, S 0.51, 10.6 FP je Fall) gegen unser U-Net 0.634, gepaart -0.228 [-0.308; -0.165]**; Betrachter `mb-arena/qc/swi_vs_sundaresan/`. Eingereiht: T2* fünf Falten [218]-[222] (~75 min je Falte); VALDO folgt, sobald die Vorverarbeitung durch ist. Erwartung vorab: Original feingetunt auf VALDO 0.30-0.45, auf T2* 0.40-0.55 -- klar unter unserer Kette, aber über dem Nachbau ohne Feintuning (0.245).
**Wiederholbarkeit:** `cmb/experiments/kern.py` führt die Kernschritte als ein Skript (`--list`, `--dry-run`, `--queue`); Lauf der CPU-Schritte 29.09. 11:05 in `logs/kern_wiederholbarkeit_0929.log`.

**Arm 7 -- Epochendosis 90 fürs Paper (Nutzerentscheid 29.09. 12:40, vorab festgelegt):** eingereiht hinter Arm 6: catalina-T2* eigenes Training 90 Ep. (Startwerte 42 und 1), Feintuning vom VALDO-Modell 90 Ep. (lr 1e-4), VALDO 90 Ep. (Startwert 42). Ergibt die Dosisreihe 60 / 90 / 120 für Training und 10 / 30 / 90 fürs Feintuning. **Erwartung:** 90 liegt zwischen 60 und 120, also -0.01 bis -0.02 gegen 60 (T2*: 0.663 / ? / 0.640; VALDO: 0.585 / ? / 0.556), Feintuning 90 gleich 30 (Kurve ab Epoche 16 flach). Regel: keine Übernahme unter +0.03 mit einem Startwert; die Reihe wird im Paper als Kurve gezeigt, nicht als Entscheidung.

**Arm 8 -- T1 und T2 als Zusatzkanäle, nur VALDO (Nutzerauftrag 29.09. 13:30; vorab festgelegt):** Obergrenze der Ein-Sequenz-Lösung (5q e). Die drei Challenge-Besten und CenSynCMB nutzen T1 + T2 + T2*. `code/zusatzkanaele_bauen.py` legt VALDOs maskierte T1/T2 (T2S-Raum; 27 der 57 CV-T1 tragen NaN -> nan_to_num) linear auf `gitter_d`, teilt im Hirn durch das 99,5-Perzentil, kappt auf [0, 1], nicht invertiert (Mittel im Hirn T1 0,41, T2 0,47). `turnier.py --zusatz t1 t2` (Sicherung `.vor-zusatz-0929`): Kanäle 3 und 4 neben T2* und FRST, Netz mit 4 Eingängen (22,50 Mio. Parameter statt 22,50 -- nur die erste Faltung wächst), Stichprobe wortgleich zur Basis (Test: Form (8, 4, 38, 62, 94), Kanalmittel 0,41 / 0,03 / 0,33 / 0,31; ohne Schalter bleibt alles bei 2 Kanälen). Läufe: Startwerte 42 und 1, 60 Epochen, gepaart gegen die Basis mit Liquor-Regel (0,646 / 0,590; Ensemble 0,647). **Erwartung:** +0,02 bis +0,05 auf VALDO, vor allem in der Precision (T1/T2 markieren Sulci, Gefäße und Kalk anders als Blutungen); Regel wie immer (beide Startwerte über der Basis, Ensemble-Intervall über Null). Nicht übertragbar auf catalina (kein T1) -- die Zahl beziffert nur, wie viel des Abstands zu CenSynCMB (74,3) an den Sequenzen liegt.

**Arm 9 -- Turnier 2: Hyperparameter und weitere Netze, 60 Epochen (Nutzerauftrag 29.09. 14:00 "umfangreiche Hyperparametersuche"; vorab festgelegt).** Feste Basis a03-aniso, `gitter_d`, 60 Ep., Startwert 42, Zwischenstandswahl; je Lauf GENAU eine Änderung. Bezug: Basis s42 0.585 roh / 0.646 mit Liquor-Regel (kreuzweise 0.622). 17 Läufe, je ~1.6 h (b48, n_patches 1600 länger), eingereiht hinter Arm 6/7/8:
| Gruppe | Läufe (Ordnerzusatz) | Bezug im Code |
|---|---|---|
| Breite | `a03-aniso-b16` (5.6 Mio.), `a03-aniso-b48` (50.6 Mio.) | Basis 32 (22.5 Mio.) |
| Netze neu | `a13-r2unet2p5d` (rekurrent-residuales 2D-U-Net auf 5 Nachbarschichten, 8.9 Mio.), `a14-mednext` (MedNeXt-S, MONAI 1.5, 5.6 Mio.), `a15-nnunet-resenc` (DynUNet mit Residual-Encoder, 16.7 Mio.) | Turnier 1: a04 2.5D 0.294, a02 nnU-Net 0.458 (e60, sauber) |
| Lernrate | `-hplr0.0001`, `-hplr0.001` | 3e-4 |
| Weight-Decay | `-hpweightdecay0`, `-hpweightdecay0.001` | 1e-4 |
| Stapel | `-hpbs2`, `-hpbs8` | 4 |
| Budget je Epoche | `-hpnpatches400`, `-hpnpatches1600` | 800 (5h: Budget je Fall ist der Engpass) |
| Läsionsanteil | `-l40`, `-l80` | 60 / 25 / 15 |
| Verlust | `-vfocal_tversky` (alpha 0.3, beta 0.7), `-vbce_dice_ohne_ds` | BCE + Dice mit tiefer Überwachung |
Code: `code/netze.py` (Sicherung `.vor-turnier2-0929`; Vorwärtspässe aller fünf neuen Netze auf (2, 38, 62, 94) geprüft, ResEnc liefert im Training drei Köpfe wie a02), `turnier.py --hp SCHLUESSEL=WERT`, `--verlust focal_tversky | bce_dice_ohne_ds`.
**Auswertung und Regel, vorab:** alle Läufe zusammen mit `operating_point` (feste Zelle + Liquor-Regel, kreuzweise als Nebenwert) gepaart gegen die Basis s42, dazu roh je Zählklasse. Mit EINEM Startwert ist nichts unter 0.08 deutbar (Abschnitt 3): Läufe mit ≥ +0.04 gegen die Basis bekommen einen zweiten Startwert; übernommen wird nur, was mit beiden Startwerten über der Basis liegt und im Zwei-Startwerte-Ensemble ein Intervall über Null hat. Erwartung: keine Änderung über +0.04 (alle bisherigen Trainingshebel lagen bei 0 oder darunter); b48 und MedNeXt am ehesten gleichauf, 2D-rekurrent deutlich darunter (~0.3 wie a04), n_patches 1600 leicht schlechter (Budget), lr 1e-3 instabil. Das Turnier dient dem Paper als Beleg der begrenzten Suche, nicht als Hoffnung auf einen Sprung.

**Arm 9, Nachtrag (Nutzerentscheid 29.09. 13:50, nach `recherche/vorgehen-modellwahl-0929.md`):** zweite Startwerte stichprobenartig auch für Nicht-Gewinner -- zwei Turnierläufe per Zufall gezogen (Python `random.seed(20260929)`, `random.sample`): **`n_patches=1600` und `bs=2`**, je `--seed 1`, eingereiht hinter dem Turnier. Zweck: die Schwelle "+0.04 -> zweiter Startwert" gilt sonst nur für Gewinner; hier wird geprüft, ob ein unauffälliger Lauf mit anderem Startwert ebenso unauffällig bleibt (Picard 2021).
**Arm 10 -- echtes nnU-Net v2 als Baseline (Nutzerentscheid 29.09. 13:50; "nnU-Net Revisited": standardisierte Baseline fehlt):** nnunetv2 2.8.1 (dl-Umgebung), Datensatz `Dataset501_VALDOCMB` aus den 57 CV-Fällen -- Eingabe die maskierten VALDO-T2S in nativer Auflösung (kein FRST, keine unserer Vorverarbeitungen; nnU-Net plant Umrechnung und Normierung selbst), Label CMB, unsere fünf Falten als `splits_final.json`; Konfiguration 3d_fullres, Trainer `nnUNetTrainer_250epochs` (Standard wären 1000; 250 aus Budgetgründen, offen gelegt), 5 Falten; Vorhersage out-of-fold, Auswertung mit unserer Läsionsmetrik nativ und aufs Gitter umgerechnet (wie Arm 6). **Erwartung:** 0.45-0.60 mit Liquor-Regel (das DynUNet-Nachbau lag bei 0.458 roh; das Framework bringt Augmentierung, Ensembles-frei, eigene Normierung), unter unserer Kette (0.646). Werkzeug `cmb/baselines/nnunet_v2.py`.

## 6 Bedrohungen der Gueltigkeit
- 57 Faelle, 139 CMB: breite Intervalle; viele Vergleiche auf denselben Falten -> Gefahr der Ueberanpassung an die CV.
- Liquor-Regel und Uebermalung 2 wurden nach Blick auf die CV-Daten entworfen (Parameter der Uebermalung ohne Label).
- Die 15 beiseitegelegten Faelle wurden am 14.09. einmal benutzt (F1 0.19, ein Fall traegt 75 % der CMB).
- Referenzannotation von drei Gruppen mit teils verschiedenen Protokollen; Inter-Rater-Wert im Paper nicht berichtet.
- Unabhaengige Bestaetigung steht aus: catalina (anderer Scanner, schwaecherer Kontrast, 5-mm-Schichten).
- **28.09.:** die 15 beiseitegelegten Faelle sind zum zweiten und letzten Mal gemessen (5ad): gepoolt 0.468 (ein Fall mit 73 Blutungen, S 0.20), ohne ihn 0.731, Median je Fall 0.667. Beide Zahlen gehoeren ins Paper; keine Wahl wurde nach diesem Blick getroffen.

## 7 Naechste Schritte
**Stand 28.09. (nach 5ac):** Trainingsrezept (5h, 5o, 5p, 5t, 5v), Intensitätsskala (5aa, 5ac), Architektur (5s), Kandidaten-Decke (5w, 5y), zweite Stufe (5r, 5z, 5ab) sind vermessen; kein Hebel liegt gesichert über der Basis. **Die Kette ist fertig:** je Kohorte Basis-U-Net (Zwei-Startwerte-Ensemble) + gemeinsames kleines 3D-CNN + Liquor-Regel auf VALDO. Was jetzt zählt, ist nicht mehr Messen, sondern Abschließen: (1) das Expertenurteil zur verblindeten Prüfseite (5b, seit 22.09. beim Nutzer) -- es entscheidet, wie viel des Restfehlers Referenz ist; (2) die EINMALIGE Messung der eingefrorenen Kette auf den 15 beiseitegelegten Fällen und nach Challenge-Metrik; (3) Paper-Zahlen als Mittel über Startwerte (5u), Grenzen aus Abschnitt 6; (4) Veröffentlichung (we6/we7, Nutzerentscheid). Einziger noch ungemessener Hebel mit ANDERER Information (Regel aus 5q): T1/T2 als Zusatzkanäle, wie bei den drei Challenge-Besten -- ein Training, wenn der Nutzer es will, aber erst nach (1).
**Stand 21.09. 21:00:** der urspruengliche Plan unten (vom 19.09., zur Nachvollziehbarkeit stehen gelassen) ist abgearbeitet bis auf grosse Ausschnitte und langes Training (eingereiht) und die einmalige Messung auf den 15 beiseitegelegten
Faellen (bewusst zuletzt). Der aktuelle Plan -- was ausgeschoepft ist, was laeuft, was offen bleibt -- steht in 5q.

Uebermalung 2 gegen keine -> zweiter Startwert der neuen Basis (gitter_d) -> Bias aus / Perzentil-Skala / z = 2 mm /
ohne FRST / Verhaeltnis / Attention / grosse Patches / 300 Epochen, alles auf gitter_d -> bestes Rezept mit zweitem
Startwert bestaetigen -> zweistufige Fassung -> Gewebekarte GM/WM/CSF/CMB -> catalina (ohne Anpassung / feingetunt /
nur-catalina als Kontrolle) -> einmalige Messung auf den 15 beiseitegelegten Faellen.

## 8 Code
Der Experimentcode (`code/`, deutsch, im Lauf der Erkundung gewachsen) wird Modul fuer Modul durch das englische Paket
`cmb/` ersetzt (Nutzerwunsch 19.09.: lesbar, strukturiert, leicht zu debuggen). Abnahmeregel: ein Modul gilt erst, wenn
es seinen Vorgaenger exakt reproduziert. Stand 19.09.: Netzbausteine und das Siegernetz portiert; mit den gespeicherten
Turnier-Gewichten sind die Ausgaben bit-identisch zum Altcode (`python -m cmb.tests.test_models_match_legacy`).
Plan und Stand je Modul: `cmb/README.md`.
**Neu 29.09.:** `cmb/review/build_slice_viewer.py` -- Schichtbetrachter ohne Server: zwei große axiale Bilder je Fall (links Referenz, rechts beliebig viele Modell-Umrisse als schaltbare Farben), Mausrad/Pfeile = z-Achse, beide Seiten synchron, Zähler TP/FP/FN je Modell und je Schicht; Nutzerwunsch. Gebaut für VALDO (`ergebnisse/qc/valdo_slices/betrachter.html`, Modelle basis / basis+liquor / kubisch / synthese2); für catalina mit privatem `--out` und den `mb-arena`-Läufen aufrufbar.
Nachtrag 29.09. 09:30: Seite ist responsiv (Handy: Fall-Auswahl oben, Wischen über dem Bild = Schicht, Knöpfe unten; Bilder untereinander im Hochformat) und hat `--single-file N` (Einzeldatei mit den N blutungsreichsten Fällen eingebettet, 480 px, JPEG 72; 24 Fälle = 29.6 MB). Die Einzeldatei mit vier Modellen wurde dem Nutzer per `schicken` (Telegram) geschickt -- nur VALDO, öffentlich. Zweiter Betrachter mit 31 VALDO-Läufen: `ergebnisse/qc/valdo_slices_alle/` (Turnier-Läufe auf `gitter_c` passen nicht auf das Bild und fehlen).

## 9 Lehre
`cmb/teaching/build_exercise.py` erzeugt eine Uebungsseite fuer die Vorlesung (eine HTML-Datei, offline, deutsch/englisch):
Studierende malen Mikroblutungen in 12 T2*-Schichten an (drei oeffentliche VALDO-Faelle, leicht/mittel/schwer), haben je
Fall drei Versuche und bekommen Dice, F1, Sensitivitaet, Precision und Spezifitaet zurueck; eigene Marken werden gruen
(Treffer) bzw. rot (Fehlalarm) gefaerbt, die uebersehenen Blutungen erscheinen nach dem dritten Versuch gelb. Die
Spezifitaet ist absichtlich dabei: sie liegt bei jedem nahe 1.0 und zeigt, warum sie bei winzigen Laesionen nichts sagt.
Die Wertungslogik ist gegen ein Handbeispiel getestet (Dice 0.444, F1 0.5). Datei: `ergebnisse/lehre/cmb_exercise.html`.

## 10 Nachvollziehbarkeit (Stand 20.09.2026) -- was fuer eine Veroeffentlichung vorliegt und was noch fehlt
**Vorhanden:** (1) `protokoll.md`: jede Messung mit Datum, Zahlen, Intervallen; (2) dieses Dokument: Begruendungskette je Eingriff,
vorab festgelegte Entscheidungsregeln und Erwartungen, negative Ergebnisse; (3) `ABWEICHUNGEN-microbleednet.md`: Unterschiede zum
Original mit Referenz-Commit 958f1cb; (4) je Trainingslauf und Falte `meta.json` (alle Rezeptparameter, Startwert, gewaehlte Epoche,
Schwelle, Dauer, VRAM) + `verlauf.json` + Modellgewichte + Wahrscheinlichkeitskarten; (5) die Warteschlangen-Datei mit jedem
Trainingsbefehl, Zeitstempel und Rueckgabewert; (6) Sicherungskopien jeder geaenderten Skriptdatei mit Datum (`dev/*.vor-*`);
(7) `docs-umgebung/`: eingefrorene conda-Umgebungen (dl, shiva), Hardware, Versionen von PyTorch, MONAI, FreeSurfer, FSL;
(8) Datenherkunft und Lizenz (`dev/lizenz-valdo.md`), feste Aufteilung (`dev/split.json`, Seed 0), Determinismus nachgewiesen
(a01 reproduziert cv5-r3 auf vier Stellen); (9) seit 20.09. ein privates Git-Repository mit Code und Dokumenten.
**Luecken, ehrlich benannt:** (a) Bis zum 20.09. gab es KEINE Versionsverwaltung; der Code-Stand je Lauf ist nur ueber datierte
Sicherungskopien rekonstruierbar. (b) Etliche Auswertungen liefen als Einmal-Skripte in der Sitzung und liegen NICHT als Datei vor:
gepaarte Bootstrap-Vergleiche, Kohorten-Aufschluesselung, Schadensbilanz der Original-Uebermalung (erste Fassung), Maskenvolumen,
Kontrastverlust durch Umrechnung, FRST-Trefferquote und -AUC, Fehleranalyse, Gittersuche Schwelle x Mindestgroesse, Nachrechnung nach
Challenge-Regel. Die ZAHLEN stehen im Protokoll. **Nachgeliefert am 20./21.09. und gegen das Protokoll geprueft (Tabelle in `REPLIKATION.md` Abschnitt 6):** exakt reproduziert sind
Schadensbilanz (`code/uebermalung2.py --pruefen dev/preproc`), Maskenvolumen (`cmb/analysis/mask_volume.py`) und Kohorten-Aufschluesselung (`cmb/analysis/cohort_breakdown.py`);
NICHT exakt reproduzierbar sind die FRST-Trefferquote (Datei: 82 % -> 96 % statt 85 % -> 96 %) und der Kontrastverlust (Datei: -26 / -15 / -30 % statt -23 / -12 / -32 %) -- der
Einmal-Code ist verloren, ab jetzt gelten die Werte aus den Dateien; die Schlussfolgerungen aendern sich nicht. (c) Literaturangaben in `recherche/` wurden von einem kleinen Lesemodell aus den Quellen gezogen und sind vor
dem Zitieren zeilenweise gegenzulesen. (d) Fehler im Original (A1-A3) sind am Quelltext gelesen, nicht ausgefuehrt.
(e) Die 15 beiseitegelegten VALDO-Faelle wurden am 14.09. einmal benutzt -- im Paper offenlegen. Am 28.09. zum zweiten und letzten Mal, mit der eingefrorenen Kette (5ad; Prüfsummen der zehn Modelldateien in `ergebnisse/validierung/heldout_0928/FROZEN.sha256`).

## 11 Begriffs- und Benennungspruefung (21.09.2026, Nutzerauftrag nach dem Fehlgriff "3D-ResNet"; fortgeschrieben 22.09.)
Geprueft: jede Bezeichnung in Dokumenten, Berichten und Docstrings gegen das, was der Code tut (`code/netze.py`, `attention_unet.py`, `frst.py`, `cv5.py`, `turnier.py`, `cmb/`), dazu Herkunft der Daten und
Masken. **Falsch oder ungenau benannt -- berichtigt:**
1. **"3D-ResNet" (zweite Stufe)** -- ist ein schlichtes 3D-CNN ohne Residualverbindungen (216 081 Parameter). Berichtigt in 5j, in den Docstrings und in `LITERATUR.md`. Zahlen unberuehrt.
2. **"60 feste Epochen"** -- fest ist der Plan, nicht der verwendete Zwischenstand: die Vorhersage stammt vom besten Zwischenstand der inneren Falte (alle 5 Epochen geprueft). Gewaehlte Epochen z. B. Basis
   40 / 60 / 50 / 35 / 60, Startwert 1: 50 / 55 / **10** / 40 / 40, catalina-T2* 55 / 60 / 20 / 55 / 35, Feintuning "30 Ep." 12 / 24 / 18 / 22 / 8. Kein Leck (die innere Falte ist nie die Testfalte), aber: (a) die
   Wahl an ~11 Faellen ist selbst eine Rauschquelle -- ein Teil der Startwert-Streuung (0.553 gegen 0.585) duerfte daher kommen; (b) Aussagen ueber "Epochen" (5h) betreffen die PLANLAENGE. Vorschlag: ein Lauf mit
   wirklich festem letztem Zwischenstand als Gegenprobe (Schalter fehlt noch).
3. **Hirnmasken catalina "aus HD-BET"** -- 58 von 75 (T2*) und 61 von 61 (SWI) Masken sind mitgelieferte Dateien der Quelle (`*_brainmask.nii.gz`; dort liegt ein Skript `make_brainmask_hdbet.py`, also sehr wahrscheinlich
   ebenfalls HD-BET, von mir nicht geprueft); selbst mit HD-BET gerechnet sind nur die 17 fehlenden T2*-Masken.
**Richtig benannt, aber leicht misszuverstehen -- Klarstellungen:**
- *a02-nnunet*: die nnU-Net-ARCHITEKTUR (MONAI `DynUNet`, Deep Supervision), NICHT das selbstkonfigurierende nnU-Net-Verfahren mit eigenem Trainingsrezept. *a06 / a08 / a09*: MONAI `BasicUNetPlusPlus`, `SwinUNETR`
  (abgebrochen), `SegResNet`. *a05*: echte Residualbloecke mit Squeeze-Excitation. *a01 / a11*: additive Attention-Gates wie bei Oktay et al. "Zehn Architekturen" = neun einstufige Netze + die zweistufige Fassung; a11 / a12 kamen spaeter dazu.
- *FRST*: echte Fast Radial Symmetry Transform nach Loy & Zelinsky, aber 2D je axialer Schicht (`code/frst.py`), nicht 3D.
- *Zero-shot*: Mittel der FUENF VALDO-Faltenmodelle -- alle anderen Strategien sind Einzelmodelle je Falte. Da ein Ensemble +0.02 bis +0.04 ueber dem mittleren Einzelmodell liegt (5j), ist der Domaenenabfall eher GROESSER als
  die gemessenen -0.041.
- *"Skala max"* (Nutzerfrage 23.09. 11:01: "was ist mit Normalisierung 0-1?") -- irrefuehrend: unsere Skala IST bereits eine 0-1-Normierung je Fall. Geliefert wird `1 - T2*/Maximum im Hirn`; die Inversion macht das hellste T2*-Voxel zu 0
  und das dunkelste zu 1. Nachgemessen an fuenf Faellen: Minimum im Hirn 1e-6 bis 1.3e-5, Maximum 0.987 bis 1.000 -- eine Min-Max-Normierung waere ein Faktor 1.000-1.013 und eine Verschiebung von einem Millionstel, also die Identitaet.
  In der Messung (5aa) liefert `minmax` deshalb exakt dieselben vier Zahlen wie `max`. Kuenftig heisst es "Min-Max je Fall auf der invertierten Skala"; "durch das Maximum" (so im Original) liest sich, als bliebe der untere Bereich ungenutzt.
- *Kontrast* hat drei Definitionen: relative Abdunklung gegen einen mm-Ring (`missed_lesions.py`, 5i / 5n), Differenz auf der invertierten Skala (`resampling_contrast.py`, 4.8; Schadensbilanz 4.1), und das Merkmal der zweiten
  Stufe (Differenz gegen einen Voxelring). Werte verschiedener Definitionen nicht vergleichen.
- *F1* ohne Zusatz = gepooltes Laesions-F1 (26er-Komponenten, Beruehrung) an der festen Zelle 0.3 / 2 mm3; daneben: inneres F1 und F1 je Falte im Trainingslog (eigene Schwelle je Falte), geschachteltes F1 der zweiten Stufe,
  VALDO-Regel (Median je Proband), F1 je Proband gemittelt (CenSynCMB). Nur gleiche Sorten vergleichen.
- *gesichert* = das 95-%-Bootstrap-Intervall ueber Faelle schliesst 0 aus; KEINE Korrektur fuer die Vielzahl der Vergleiche. **Nachtrag 22.09.:** dieses Intervall enthaelt das TRAININGSrauschen nicht -- zwei Laeufe desselben Rezepts mit
  anderem Startwert liegen bis 0.083 auseinander und werden als "gesichert verschieden" ausgewiesen (5u). Ab 22.09. gilt: Einzel-Startwert-Effekte unter ~0.08 sind Richtungsangaben, keine Beweise (Abschnitt 3).
- *Basis* (Nutzerfrage 22.09. 21:25) -- ich habe das Wort in ZWEI Bedeutungen benutzt: (a) das **Basisrezept**, also die Methode (a03-aniso auf gitter_d, T2* + FRST, 60-Epochen-Plan mit Zwischenstandswahl, 800 Ausschnitte
  38 x 62 x 94, feste Nachbearbeitung + Liquor-Regel); (b) den **Basis-Lauf s42** (`ergebnisse/turnier/a03-aniso-e60-gd`, Startwert 42, 0.585 / 0.646), der in allen gepaarten Tabellen der Bezug ist, weil er zuerst fertig war.
  Das ist gefaehrlich, seit 5u bekannt ist, dass s42 der beste von drei Startwerten ist: "Basis = 0.646" liest sich als Eigenschaft der Methode, ist aber die Eigenschaft eines Lauf-Loses. **Sprachregelung ab hier:** "Basisrezept"
  fuer die Methode (zu berichten als 0.605 +- 0.034, ausgeliefert wird das Zwei-Startwerte-Ensemble 0.647), "Basis-Lauf s42" fuer den Vergleichspartner. In Tabellen bleibt s42 der Bezug -- gepaart wird paarweise verglichen --,
  aber ein Abstand unter 0.08 gegen genau diesen Lauf ist keine Aussage ueber das Rezept.
- *harte Negative*: Schuerfen in der STICHPROBE des U-Net; bei Dou et al. 2016 lernt dagegen die zweite Stufe auf den Fehlalarmen der ersten -- verwandt, nicht dasselbe. *blob loss*: Tagungsband nicht geprueft (arXiv 2205.08209).
**Ohne Befund:** Verluste (BCE+Dice, blob, fnrw, Zentrum tun, was die Namen sagen; Abweichungen vom Paper stehen in den Docstrings), Liquor- und Gewebeklassen (FreeSurfer-Tabelle), Biaskorrektur (`fast -B`), Gitter,
Stichprobenanteile, AdamW / Kosinus / InstanceNorm, Ensemble (Mittel der Karten), Spiegel-TTA (acht Ansichten), geschachtelte Auswertung der zweiten Stufe, oeffentliche QC-Seite und ihr README (keine der Fehlbenennungen).

