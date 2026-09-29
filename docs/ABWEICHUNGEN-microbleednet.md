# Abweichungen vom Original-MicrobleedNet (Sundaresan et al.)

**Zweck (Nutzerauftrag 19.09.2026):** jederzeit sagen koennen, WAS bei uns anders ist als im Original und WIE STARK.
Diese Datei wird bei JEDER Aenderung an Vorverarbeitung, Netz, Training, Nachbearbeitung oder Metrik fortgeschrieben
(neue Zeile oder geaenderte Zeile mit Datum). Staerke: **keine** (identisch) / **klein** (Reparatur, aendert das
Verfahren nicht) / **mittel** (gleiche Idee, andere Parameter) / **gross** (anderes Verfahren).

## Referenzstand des Originals

- Original: https://github.com/v-sundaresan/microbleed-detection , Stand **Commit 958f1cb vom 19.03.2026**
  ("Merge pull request #7 from g-shanker/main"). Am 19.09.2026 online geprueft: das ist der neueste Commit dort.
- Lokale Kopie: `~/software/microbleed-detection` = Fork `github.com/mendeltem/custom_microbleednet`,
  = Original 958f1cb + 1 Fork-Commit + nicht eingecheckte Aenderungen in 3 Dateien (Schicht A unten).
- Gelesen am Quelltext: `microbleednet/scripts/data_preparation.py` (Vorverarbeitung). Alles, was unten mit
  "(Notiz)" markiert ist, stammt aus meinen Projektnotizen vom 11.09. und ist NICHT heute am Original nachgelesen.

## Schicht A -- unser Fork `custom_microbleednet` gegen das Original (alles hier ist UNSERE Aenderung)

Nutzer 19.09.: die Aenderungen im Fork gehoeren zu unserem Beitrag und damit in den Vergleich. Vollstaendig
aufgeschluesselt mit `git diff 958f1cb 0aa3acf` (Fork-Commit vom 13.07.2026) und `git diff` (nicht eingecheckt).
"Am Quelltext gelesen" heisst: Aufruf und Signatur verglichen, NICHT ausgefuehrt.

| Nr | Datei | Original (958f1cb) | Fork | Staerke | Warum / Beleg | Stand |
|---|---|---|---|---|---|---|
| A1 | `commands.py`, Aufruf des CDisc-Feintunings | `main(..., aug=..., save_cp=..., save_wei=..., model_dir=...)` | `perform_augmentation=, save_checkpoint=, save_weights=, ...` | klein (Reparatur) | `main()` kennt die Namen `aug`/`save_cp`/`save_wei` nicht (Signatur: `perform_augmentation, save_checkpoint, save_weights, ...`) -> im Original ist ein TypeError zu erwarten, das CDisc-Feintuning per Kommandozeile startet nicht. Am Quelltext gelesen. | eingecheckt 0aa3acf |
| A2 | `cdet_finetune_function.py`, Datensaetze | `CDetPatchDataset(pos, neg, perform_augmentations=True)` | `CDetPatchDataset(pos, neg, ratio='1:1', ...)` | klein (Reparatur) + **mittel (Festlegung)** | `__init__(minority, majority, ratio, ...)` verlangt `ratio` -> Original-Aufruf ohne `ratio` = TypeError. Mit der Reparatur ist das Verhaeltnis positiv:negativ auf **1:1** festgelegt (moeglich waeren auch '2:1', 'random'). Am Quelltext gelesen. | eingecheckt |
| A3 | `cdisc_finetune_function.py`, Datensaetze | `CDiscPatchDataset(NEGATIVE, POSITIVE, ...)`, ohne `ratio` | `CDiscPatchDataset(POSITIVE, NEGATIVE, ratio='1:1', ...)` | **mittel** | Signatur ist `(minority_class, majority_class, ratio)`: im Original sind die Klassen vertauscht uebergeben und `ratio` fehlt. Am Quelltext gelesen. | eingecheckt |
| A4 | `cdisc_finetune_function.py`, Einfrieren | `freeze_layers_for_finetuning` kennt nur die U-Net-Schichtnamen; der Klassifikator-Kopf (classconvfirst, classdown1/2, fc1-3) bleibt IMMER eingefroren | Kopf wird ausdruecklich aufgetaut; nicht eingecheckt zusaetzlich `inpconv` | klein (Fehlerkorrektur), Wirkung gross | Ohne das kollabiert CDisc beim Feintuning auf "alles negativ" (Genauigkeit 0.5, 0 Positive) -- beobachtet in der MB-Arena. | Kopf eingecheckt, `inpconv` nicht |
| A5 | `utils.py` | `load_model` vorhanden | zweite Definition `load_model(checkpoint_path, model, mode)` am Dateiende angehaengt | klein | ueberschreibt die erste; ob sich das Verhalten unterscheidet, ist noch NICHT verglichen (offen). | eingecheckt |
| A6 | `cdet_evaluate_function.py` | -- | zwei `print(model path)` | keine | Diagnoseausgabe | eingecheckt |
| A7 | `cdet_evaluate_function.py`, Kandidatenschwelle | `predictions > 0.2` | `> 0.1` | mittel | Vortrainierte Gewichte erreichen auf catalina-SWI hoechstens 0.217 je Fall; bei 0.2 in 59/61 Faellen KEINE Kandidaten, bei 0.1 in 54/61 (Mittel 2.75 je Fall). Gemessen 07.09. (diagnose.json). | NICHT eingecheckt |
| A8 | `data_preparation.py`, Hirnmaske (3 Stellen) | `brain_mask = (image > 0)` | `binary_fill_holes(image > 0)` | klein | Voxel mit Wert 0 IM Hirn (sehr dunkle CMB/Gefaesse) fielen aus der Maske und wurden beim Maskieren geloescht. | NICHT eingecheckt |
| A9 | Repo-Hygiene | Build-Artefakte, egg-info, Caches, `scripts-3.10/` im Repo | entfernt, `.gitignore` | keine | 9645 Zeilen geloescht, kein Verhalten betroffen | eingecheckt |

**Einordnung fuer die Arbeit:** A1-A3 bedeuten, dass der veroeffentlichte Feintuning-Pfad des Originals (Stand 958f1cb)
nach dem Quelltext nicht ohne Eingriff laeuft; A4 bedeutet, dass er -- einmal lauffaehig -- den Klassifikator nicht
trainiert. Das ist ein eigener Befund ueber das Original, unabhaengig von unseren VALDO-Netzen. Vor einer Aussage nach
aussen: A1-A3 einmal AUSFUEHREN (frischer Klon von 958f1cb, Aufruf wie in den Beispielskripten) und die Fehlermeldung
woertlich festhalten.

Die Funktion `inpaint_vessels` (Gefaess-Uebermalung) ist in der lokalen Kopie **unveraendert** gegenueber 958f1cb --
`frangi(sigmas=(0.5,1.2,0.2), alpha=0.9, beta=20)`, Strukturtensor-Linearitaet, KMeans k=2 je Schicht, Rettung bei
eccentricity < 0.9 und solidity > 0.5 stammen also aus dem Original.

## Schicht B -- unser VALDO-Verfahren gegen das MicrobleedNet-Verfahren

| Baustein | Original MicrobleedNet | Bei uns (Stand 19.09.2026) | Staerke | Warum / gemessen | Datei |
|---|---|---|---|---|---|
| Bias-Korrektur | nicht Teil der Python-Kette (FSL-Vorverarbeitung per Shell-Skript, Notiz) | FSL FAST -B vor der Kette, 72/72 | klein | Std im Hirn Kohorte 1 etwa -12 %, Kohorte 2 fast 0 | `code/preprocess.py` |
| Hirnmaske | Bild > 0 | gelieferte VALDO-Maske (Bild > 0, Loecher gefuellt). SynthSeg-Hirn + 2 Voxel (`--maske synthseg`) am 19.09. geprueft: F1 0.421 gegen 0.470, NICHT uebernommen | klein | VALDO-Maske ist in Kohorte 3 zu 40 % Nicht-Hirn (2533 ml), Kohorte 2 16 % | `code/turnier.py` |
| Gefaess-Uebermalung | `inpaint_vessels` (s. o.) | bisher identisch uebernommen (gitter_c). **In Pruefung:** keine (gitter_d) bzw. Uebermalung 2 (gitter_e/f/g): beta 0.5, Skalen in mm, eine Schwelle je Volumen, nur >= L mm lange duenne Komponenten | bisher keine -> gross | Original trifft 109/236 VALDO-CMB, 17 danach unsichtbar, uebermalt 12-17 % des Hirns; Uebermalung 2 (3 %, 8 mm): 1/236, 0.9-1.5 %. **Training ohne Uebermalung: F1 0.591 gegen 0.455 mit Original-Uebermalung, gepaart +0.136 [+0.065; +0.226] (19.09., 4/5 Falten) -> Original-Uebermalung gestrichen**; Uebermalung 2 gegen "keine" laeuft | `code/uebermalung2.py` |
| Invertierung / Skala | `1 - Bild/Maximum`, maskiert | identisch. z-Normierung je Fall (`--norm z`) am 19.09. im Paket mit Augmentierung geprueft: F1 0.410 gegen 0.470; einzeln 28.09. ausgewertet: VALDO -0.005, Zero-shot T2* -0.004 / SWI -0.022, alle n. s. Landmarken-Angleich (`--norm landmarken`, Nyul-Udupa, Referenz `dev/landmarken_valdo.json`): VALDO -0.064, Zero-shot T2* -0.048 / SWI -0.034 (beide gesichert), gemischt VALDO-Anteil 0.355 gegen 0.582. Beides NICHT uebernommen (ARBEIT.md 5ac) | keine | Skala zwischen Kohorten nicht harmonisiert (5.-95. Perzentil 0.25-0.74 gegen 0.65-0.98) | `code/turnier.py` |
| FRST | Code (gelesen 19.09.): `fast_radial_symmetry_xfm` je Schicht, `radii = [2, 3]`, alpha 2, auf dem Originalgitter, min-max-normiert. PAPER 2023: Mittel ueber vier Radien 2/3/4/6 -- Code und Paper weichen ab | eigene FRST (`code/frst.py`, Loy & Zelinsky, stetig), Radien [2,3,4,6] Voxel = 1-3 mm auf dem gemeinsamen Gitter, 99.9-Perzentil-Normierung | mittel | Top-1-%-Treffer an der CMB 82 % -> 89 % (11.09.); ohne Uebermalung 96 % (19.09.) | `code/frst.py`, `code/gitter_c_bauen.py` |
| Gitter | Originalaufloesung je Fall | alle Faelle auf 0.5 x 0.5 x 1.0 mm, auf das Hirn zugeschnitten (Gitter C) | gross | drei Kohorten mit 0.45x0.45x4 / 0.49x0.49x0.8 / 1x1x3-4 mm in EINEM Netz | `code/gitter_c_bauen.py` |
| Netz | zweistufig: CDet (Kandidaten, 48^3) + CDisc (Klassifikator, 24^3), Lehrer-Schueler-Destillation (Notiz) | einstufig: anisotropes 3D-U-Net `a03-aniso` (erste Stufe nur in der Schicht), 22.5 Mio Parameter; Turnier ueber 10 Architekturen | gross | a03 F1 0.404 gegen Attention-U-Net 0.245, gepaart +0.159 [+0.091; +0.234]; zweistufige Fassung am 19.09. gebaut und geschachtelt geprueft (`cmb/stage2/`): 0.599-0.616 gegen 0.627 ohne zweite Stufe -- nicht uebernommen (zu wenige Kandidaten) | `code/netze.py` |
| Eingangskanaele | Bild + FRST | identisch (T2* + FRST). Kontrolle ohne FRST (20.09.): F1 0.535 gegen 0.588, gepaart -0.053 [-0.116; +0.004] -> FRST bleibt | keine | FRST hilft als Wegweiser, trennt aber Kandidaten nicht (AUC 0.56) | `code/cv5.py` |
| Patch / Stichprobe | 48^3 nicht ueberlappend, klassenweise (Notiz) | 38x62x94 Voxel (38x31x47 mm), 800 je Epoche, 60 % mit CMB (zufaellige Lage im Patch) / 25 % Hirn / 15 % beliebig; 1:2 kollabiert (5g); Sundaresan-nahe 24x24x24 mm am 22.09. geprueft: F1 0.369 gegen 0.585, gepaart -0.216 [-0.277; -0.156], NICHT uebernommen; 48-mm-Wuerfel: 0.569, -0.016 n. s., nicht uebernommen -- das Rezept liegt am Optimum (ARBEIT.md 5t) | gross | zentrierte CMB -> Netz lernte "Laesion = Mitte" (11.09.) | `code/cv5.py` |
| Verlust | gewichtete Kreuzentropie + Dice mit Pixelgewichten (Notiz) | BCE + Dice, ungewichtet | mittel | einziges Rezept, das am 11.09. messbar lernte | `code/cv5.py` |
| Trainingsdauer | -- (nicht nachgelesen) | bis 19.09.: <= 40 Epochen mit Fruehstopp; jetzt 60 feste Epochen; 300 eingereiht | -- | 40 -> 60 fest: F1 0.404 -> 0.470 roh; Fruehstopp liess eine Falte bei Epoche 5 stehen | `code/turnier.py --epochen` |
| Augmentierung | eigenes Modul `augmentations` (Notiz) | nur Spiegelungen. Drehung/Massstab/Kontrast/Rauschen (`--aug`) am 19.09. geprueft, nicht uebernommen | mittel | Paket mit z-Normierung: -0.060 [-0.135; +0.011], alle 5 Falten niedriger | `code/turnier.py` |
| Nachbearbeitung | Abstand zum Hirnrand >= 8 Voxel, Formfilter je Schicht (eccentricity > 0.4 und solidity < 0.5 -> weg), < 2 Voxel weg (Notiz) | Schwelle + Mindestgroesse ueber die innere Falte, Klumpen-Waechter (> 2100 Voxel weg), **Liquor-Regel** (Mehrheitslabel SynthSeg-Liquor -> weg) | gross | Liquor-Regel +0.05 bis +0.09 F1 ohne TP-Verlust; ein Randabstand wie im Original wurde bei uns nie gemessen (offen) | `code/cv5.py`, `code/gewebekarte.py` |
| Lehrer-Schueler | CDisc wird per Wissensdestillation aus einem Lehrer-Netz trainiert (Notiz; weiche Etiketten statt harter 0/1) | **NICHT geprueft** (Stand 22.09.2026). Unser gemeinsames 3D-CNN lernt direkt aus harten Etiketten. Offene Luecke, vom Nutzer am 22.09. erfragt; naheliegend waere die Wahrscheinlichkeitskarte der ersten Stufe als weiches Ziel -- passt zur Datenarmut auf VALDO (278 Kandidaten) | offen | -- | -- |
| Metrik | -- | 26er-Komponenten, Beruehrungsregel (`metric.py`); zusaetzlich offizielle VALDO-Regel (`metric_valdo.py`) | -- | beide werden berichtet | `code/metric.py` |
| Auswertung | -- | 5-fach-CV mit innerer Falte, 15 Faelle beiseite | -- | -- | `dev/split.json` |

## Offen / noch nachzulesen am Original
- Verlust, Patch-Strategie, Augmentierung, Nachbearbeitung und Trainingsdauer des Originals stehen hier aus meinen
  Notizen, nicht aus dem heutigen Quelltext -- beim naechsten Anfassen am Code pruefen und "(Notiz)" streichen.
- A1-A3 am frischen Klon des Originals AUSFUEHREN und die Fehlermeldungen woertlich festhalten; A5 (doppeltes `load_model`) vergleichen.
- A7, A8 und der `inpconv`-Teil von A4 sind nicht eingecheckt; ohne Commit gehen sie bei einem frischen Klon verloren.

## Aenderungsprotokoll dieser Datei
- 19.09.2026 angelegt (Claude). Referenzstand 958f1cb online bestaetigt.
- 19.09.2026 Schicht A vollstaendig aufgeschluesselt (Nutzer: die Fork-Aenderungen sind unser Beitrag): A1-A9.
- 19.09.2026 z-Normierung + Augmentierung geprueft und verworfen (Zeilen Invertierung/Skala, Augmentierung).
- 19.09.2026 FRST-Zeile: Radien am Code gelesen, Abweichung Code gegen Paper vermerkt. Vergleich Sundaresan/VALDO: recherche/sundaresan-gegen-valdo.md.
- 19.09.2026 15:30 Original-Uebermalung gestrichen (+0.136 gesichert ohne sie); enge Eingangsmaske geprueft und verworfen.
- 19.09.2026 abends: zweite Stufe geprueft und verworfen (Zeile Netz); Nachbearbeitung: gemeinsame Schwelle 0.3 + Mindestgroesse 2 mm3, kreuzweise 0.633.
- 20.09.2026 02:40: FRST bestaetigt (ohne: -0.053), Gitter z = 1 mm bestaetigt (z = 2 mm: -0.036); Bias-Korrektur Zwischenstand -0.058 ohne.
- 20.09.2026 14:35: Verlust -- Original Kreuzentropie (+ Destillation), bei uns BCE+Dice; NEU als Option `--verlust blob` (ein Term je Blutung, Kofler 2023), Lauf eingereiht, noch ohne Ergebnis (Staerke: offen). Ortsanalyse (`cmb/analysis/fp_location.py`): Original hat keinen Ortsfilter, wir nur die Liquor-Regel; weitere Ortsregeln bringen gemessen nichts (ARBEIT.md 5i).
- 20.09.2026 abends: Verlust/Ausgabe -- laesionsweiser Verlust gemessen (VALDO: Sensitivitaet kleiner Blutungen 0.52 -> 0.64, F1 -0.044 n. s., Staerke: offen, catalina laeuft); NEU Zentrumskarte als Hilfsausgabe (`a12-anisozentrum`, `--verlust zentrum`, nach CenSynCMB 2026), Lauf eingereiht. Zweite Stufe: Original zweistufig mit Destillation; bei uns auch auf 961 catalina-Kandidaten kein Gewinn gegen die reine Schwelle (ARBEIT.md 5j). Inferenz: Spiegel-TTA und Startwert-Ensemble als Werkzeuge (`cmb/analysis/tta_predict.py`, `ensemble.py`); Original hat beides nicht.
- 20.09.2026 spaet: Training -- NEU als Optionen `--verlust fnrw` (Neugewichtung nach Uebersehenen, CenSynCMB) und `--harte-negative` (Online-Schuerfen eigener Fehlalarme; das Original schuerft nicht, seine zweite Stufe lernt auf den Kandidaten der ersten); Laeufe eingereiht, Staerke: offen. Inferenz: Spiegel-TTA gemessen -- VALDO +0.02, catalina +-0 (Staerke: schwach).
- 20.09.2026 22:55: Zentrumskarte als Hilfsausgabe gemessen -- F1 -0.054 [-0.110; -0.006] an der festen Zelle, Sensitivitaet kleiner Blutungen 0.52 -> 0.66, Fehlalarme 86 -> 143; wie der laesionsweise Verlust empfindlicher, aber nicht besser (Staerke: negativ; nicht uebernommen). Harte Negative vor fnrw gezogen.
- 21.09.2026 frueh: harte Negative (+0.000) und Neugewichtung nach Uebersehenen (-0.030 n. s.) gemessen, nicht uebernommen; blob zweiter Startwert -0.063, blob auf SWI gleichauf. NEU als Option: Gewebekarte (SynthSeg) als fuenf Eingabekanaele (`--gewebe`); das Original kennt keine Anatomie-Eingabe. Laeufe eingereiht, Staerke: offen.
- 21.09.2026 07:10: Stichprobe/Verlust des Originals AN DER QUELLE geprueft (Nutzerfrage "warum funktioniert bei Sundaresan 1:x?"): im Code (Commit 958f1cb, `cdet_train_function.py` Z.80, `cdisc_train_function.py` Z.123, `datasets.py`) stehen beide Stufen auf `ratio='1:1'` (ein Ausschnitt mit CMB je Ausschnitt ohne); die Klasse kennt sonst nur '2:1' (ZWEI positive je negativem) und 'random' -- mehr Negative als Positive gibt es im Original nicht. Das Paper (Frontiers Neuroinf. 2023) nennt kein Verhaeltnis; das Ungleichgewicht faengt es im VERLUST ab: CE + Dice mit 10-facher Gewichtung der CMB-Voxel; Ausschnitte 48^3 (Detektion) bzw. 24^3 (Diskriminator), Stapel 8, 100 Epochen mit Fruehstopp (Geduld 20), Augmentierung x10 / x5. Bei uns: 60 % Ausschnitte mit CMB (1.5:1), keine Voxelgewichtung; 1:2 kollabiert (ARBEIT.md 5g). Damit ist die Tabellenzeile "Patch / Stichprobe" verifiziert.
- 22.09.2026 07:43: Zeile Patch/Stichprobe -- Sundaresan-nahe 24-mm-Ausschnitte gemessen und verworfen (-0.216 gesichert; ARBEIT.md 5t).
- 22.09.2026 10:17: 48-mm-Ausschnitte gemessen: -0.016 n. s., nicht uebernommen; Dosisreihe 24 / Rezept / 48 mm abgeschlossen (ARBEIT.md 5t).
- 22.09.2026 22:26: Zeile Lehrer-Schueler ergaenzt (nie geprueft, Nutzerfrage); QC-Seite mit 3 Spalten fuer catalina.
- 28.09.2026 Landmarken- und z-Normierung einzeln ausgewertet (Laeufe 23./24.09.): beide nicht uebernommen; die Skala des Originals bleibt (Zeile Invertierung/Skala, ARBEIT.md 5ac).
