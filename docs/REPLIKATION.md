# Beste Strategie und Replikation -- Stand 23.09.2026, 06:56

Zweck: EIN Dokument, aus dem hervorgeht, (1) was nach allen Messungen bis heute die beste Strategie ist -- fuer VALDO und fuer
catalina --, (2) auf welchem Weg wir dorthin gekommen sind, und (3) mit welchen Befehlen und Dateien sich jede Zahl nachrechnen
laesst. Begruendungen stehen in `ARBEIT.md` (Abschnitte in Klammern), Rohmessungen in `protokoll.md`, private Summen in
`/home/uchralt/data/work/mb-arena/protokoll.md`. catalina erscheint hier nur als Summen; die Daten verlassen Alita nicht.

Alle Befehle laufen in `/home/uchralt/data/work/valdo-t2s` mit `PY=/home/uchralt/miniconda3/envs/dl/bin/python`.
Auswertungen brauchen nur die CPU (`CUDA_VISIBLE_DEVICES="" nice ...`), Trainings laufen ueber die Warteschlange
(`grossauftrag --einreihen --vram 20 --gib 14 -- <Befehl>`), nie zwei gleichzeitig.

---

## 0 Kurzfassung: die beste Strategie

| | VALDO (T2*-GRE, oeffentlich, 57 CV-Faelle, 139 CMB) | catalina-T2* (privat, 75 Faelle, 828 CMB) | catalina-SWI (privat, 61 Faelle, 652 CMB) |
|---|---|---|---|
| Vorverarbeitung | FSL-FAST-Biaskorrektur -> Hirnmaske -> Skala 1 - I/max (das ist bereits eine Min-Max-0-1-Normierung je Fall, siehe `ARBEIT.md` 11) -> **keine** Gefaess-Uebermalung -> Gitter 0.5 x 0.5 x 1.0 mm, hirnbeschnitten -> FRST (Radien 2, 3, 4, 6 Voxel) als zweiter Kanal | dieselbe Kette; Hirnmasken ueberwiegend mit den Quelldaten geliefert (dort per HD-BET-Skript erzeugt, ungeprueft), 17 fehlende selbst mit HD-BET | dieselbe Kette |
| Netz und Training | anisotropes 3D-U-Net `a03-aniso`, 60-Epochen-Plan ohne Fruehstopp (verwendet wird der beste Zwischenstand der inneren Falte, alle 5 Epochen geprueft), AdamW 3e-4 + Kosinus, BCE+Dice, 800 Ausschnitte je Epoche (38 x 62 x 94), 60 % Laesion / 25 % Hirn / 15 % beliebig, Spiegelungen | **eigenes Training** mit demselben Rezept ODER Feintuning der VALDO-Faltenmodelle (30 Ep., lr 1e-4) -- gleichwertig; 120 Ep. sind gesichert schlechter (Schritt 25); gemischt mit VALDO bei 120 Ep. gleichauf mit eigenem 120, aber -0.02 (n. s.) unter eigenem 60 | **eigenes Training**, KEIN Start vom T2*-Netz |
| Vorhersage | Schiebefenster; Mittel aus zwei Startwerten (hebt ein durchschnittliches Modell verlaesslich auf das Niveau des besten); Spiegel-TTA optional (+0.02) | ein Modell; Ensemble / TTA ohne gesicherten Effekt | ein Modell |
| Nachbearbeitung (fest) | Schwelle 0.3, >= 2 mm3, Komponenten > 2100 Voxel weg, **Liquor-Regel** (SynthSeg-Mehrheitslabel) | dieselbe; Liquor-Regel mit SynthSeg `--robust` direkt auf T2* | dieselbe, aber **OHNE Liquor-Regel**: gemessen 23.09. kostet sie hier 0.020 (0.614 gegen 0.634) -- 74 echte Blutungen tragen ein Liquor-Label |
| Zweite Stufe (seit 21.09.) | EIN kleines 3D-CNN fuer alle Kohorten auf Kandidaten ab Wahrscheinlichkeit 0.15 (Abschnitt 3a): gleichauf mit der Liquor-Regel (0.625-0.636 gegen 0.627) | gleichauf (0.672 gegen 0.670) | **gesichert besser: 0.670 gegen 0.626 (+0.044 [+0.012; +0.085]), mit unabhaengiger erster Stufe wiederholt: 0.662 gegen 0.617 (+0.045 [+0.018; +0.085])**; die harte Liquor-Regel SCHADET hier (74 echte Blutungen tragen ein Liquor-Label) |
| Laesions-F1 (gepoolt, out-of-fold) | 0.585 roh / **0.646** mit Liquor-Regel (Startwert 42); 0.553 / 0.590 (Startwert 1); 0.501 / 0.578 (Startwert 2); **Mittel ueber drei Startwerte 0.546 roh / 0.605 mit Regel (Spanne 0.07-0.08)** -- das ist die zu berichtende Zahl EINES Modells; Zwei-Startwerte-Ensemble 0.594 / **0.647** (ausgeliefertes Modell), drei Mitglieder 0.581 / 0.636 | **0.663** roh / **0.675** mit Liquor-Regel; Feintuning 0.657; ohne die drei Extremfaelle 0.696 | **0.634** roh (mit Liquor-Regel 0.614 -- schlechter, siehe oben); mit gemeinsamer zweiter Stufe 0.670 |
| weitere Masse | VALDO-Regel Median je Proband 0.667; F1 je Proband gemittelt 0.571 (nur Probanden mit Blutung) bzw. 0.626 (alle) | Sensitivitaet 0.66, Precision 0.67, 3.6 Fehlalarme je Fall | Sensitivitaet 0.65, Precision 0.62, 4.3 Fehlalarme je Fall |

Nicht Teil der besten Strategie, weil gemessen ohne Gewinn: Gefaess-Uebermalung (Original und eigene), z-Normierung + Augmentierung,
enge Eingangsmaske, zweite Stufe JE KOHORTE (vier Klassifikatoren; die GEMEINSAME zweite Stufe ist seit dem 21.09. Teil der Kette, 3a), laesionsweiser Verlust, Zentrumskarte, Mischen der Kohorten bei gleichem
Budget, Zero-shot. Einzelheiten in Abschnitt 5.

**Wo wir stehen:** unter der Metrik der neuesten Arbeit der VALDO-Organisatoren (CenSynCMB, arXiv 2607.05325: 74.3 mit T1 + T2 + T2*)
liegen wir bei 0.57-0.63 -- im Bereich ihrer Vergleichsverfahren (61-66). Der Abstand steckt in der Precision; unser Engpass ist die
Trennung von Blutung und Verwechsler mit nur EINER Sequenz (`ARBEIT.md` 5j, 5k).

---

## 1 Voraussetzungen

- Rechner und Software: `docs-umgebung/system.txt` (RTX A5000, i9-12900K, 62 GB; torch 2.6.0+cu124, monai 1.5.2, numpy 2.3.5, scipy 1.18.1,
  nibabel 5.4.2, scikit-learn 1.9.0; FreeSurfer 8.2.0; FSL 6.0.7.23); eingefrorene Umgebungen `docs-umgebung/` (conda `dl`, `shiva`).
  HD-BET liegt in der Umgebung `medizin`. Umgebungen nie mischen.
- Daten VALDO 2021 Task 2: `/home/uchralt/data/extern/valdo2021/Task2` (72 Faelle, 236 CMB; Lizenz CC BY-NC-SA 4.0, `dev/lizenz-valdo.md`).
- Daten catalina: `~/data/sourcedata/catalina` (privat). Referenz-Original: MicrobleedNet Commit 958f1cb; lokale Kopie `~/software/microbleed-detection`
  ist unser Fork (`ABWEICHUNGEN-microbleednet.md`, Schicht A).
- Code: `code/` (Trainingsgeruest, deutsch, unveraendert benutzt), `cmb/` (sauberes englisches Paket: Auswertung, Transfer, zweite Stufe, QC).
  Versionsstand: privates Repository `cmb-t2star-pipeline`, taeglicher Schnappschuss 23:40 (`github-cmb-sync.sh`); vor dem 20.09. nur datierte
  Sicherungskopien `*.vor-*`.

---

## 2 VALDO -- Schritt fuer Schritt

| Nr. | Schritt | Befehl | Ergebnis / Pruefzahl |
|---|---|---|---|
| V1 | Aufteilung | `$PY code/split.py` (Seed 0) | `dev/split.json`, `dev/cases.json`: 15 Faelle beiseite (97 CMB), 57 in 5 Falten (139 CMB) |
| V2 | Biaskorrektur + Originalkette | `$PY code/preprocess.py --out dev/preproc --all` | `dev/preproc/<id>/` mit `fast_restore.nii.gz` (FSL FAST -B) |
| V3 | Eingang OHNE Uebermalung | `$PY code/preproc_roh_bauen.py` | `dev/preproc_roh/` (Bild = 1 - fast_restore / max in der Maske) |
| V4 | Trainingsgitter | `GITTER_C_QUELLE=dev/preproc_roh GITTER_C_ZIEL=dev/gitter_d $PY code/gitter_c_bauen.py` | `dev/gitter_d/<id>/<id>_{image,frst,label}.nii.gz`, 72/72; Komponentenzahl Quelle = Ziel je Fall geprueft |
| V5 | Fallliste | `$PY -m cmb.transfer.build_valdo_manifest --grid dev/gitter_d --out dev/manifest_valdo_gitter_d.json` | 57 Faelle, 139 Laesionen; am 20.09. nachgebaut und mit der benutzten Datei verglichen: identisch |
| V6 | Gewebekarte | `$PY code/we5.py --synthseg` (mri_synthseg auf T1 im T2*-Raum, auf das Gitter umgerechnet; 33 von 72 T1 tragen NaN -> vorher `nan_to_num`, sonst liefert SynthSeg lautlos ein leeres Bild) | `dev/synthseg/<id>_synthseg.nii.gz`, 72/72 |
| V7 | Training | `$PY code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d` (Startwert 42; zweiter Startwert: `--seed 1`) | `ergebnisse/turnier/a03-aniso-e60-gd/fold<k>/`: `modell.pt`, `meta.json`, `verlauf.json`, `<id>_pred_proba.nii.gz` (out-of-fold); je Falte ~21 min |
| V8 | Auswertung roh | `$PY -m cmb.transfer.evaluate --run basis=ergebnisse/turnier/a03-aniso-e60-gd --manifest dev/manifest_valdo_gitter_d.json` | F1 0.585, P 0.52, S 0.67, 1.51 Fehlalarme je Fall |
| V9 | Auswertung mit Liquor-Regel, fester Arbeitspunkt | `$PY -m cmb.analysis.operating_point --run basis=... --manifest ... --synthseg "dev/synthseg/{id}_synthseg.nii.gz"` | feste Zelle 0.3 / 2 mm3: **0.646** (kreuzweise gewaehlt 0.622) |
| V10 | Ensemble aus zwei Startwerten | `$PY -m cmb.analysis.ensemble --runs ergebnisse/turnier/a03-aniso-e60-gd ergebnisse/turnier/a03-aniso-s1-e60-gd --out ergebnisse/turnier/ens2-a03-aniso-e60-gd`, dann V8/V9 | 0.594 roh, **0.647** mit Liquor-Regel, 0.82 Fehlalarme je Fall |
| V11 | Spiegel-TTA (GPU, ~25 min) | `$PY -m cmb.analysis.tta_predict --run ergebnisse/turnier/a03-aniso-e60-gd --net a03-aniso --grid dev/gitter_d --out ergebnisse/turnier/a03-aniso-e60-gd-tta` | 0.604 roh (gepaart +0.019 [-0.017; +0.058]), 0.652 mit Liquor-Regel |
| V12 | Mass je Proband (Challenge / CenSynCMB) | `$PY -m cmb.analysis.subject_level_f1 --run ... --manifest ... --synthseg ...` | Median 0.667; Mittel 0.571 / 0.626 |
| V13 | Fehlerbild | `$PY -m cmb.analysis.fp_location ...`, `$PY -m cmb.analysis.missed_lesions ...` | Fehlalarme nach Ort; Sensitivitaet klein / mittel / gross 0.52 / 0.69 / 0.79 |
| V14 | Pruefseiten | `$PY -m cmb.review.build_qc ...`, `$PY -m cmb.review.build_review ...` | `qc.html` (zwei Spalten), Kandidaten-Pruefseite (201 Kandidaten) |
| V15 | verblindete Pruefseite fuer die Etikett-Decke | `$PY -m cmb.review.build_review --blind --fixed-cell --repeat 20 --seed 0 --predictions ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d --out ergebnisse/review/a03-aniso-e60-gd-blind --dataset VALDO --model a03-aniso-e60-gd`; Auswertung `$PY -m cmb.review.summarise_review <Urteile>.json --key ergebnisse/review/a03-aniso-e60-gd-blind.key.json` | 54 Fehlalarme + 47 verpasste + 94 Treffer-Komponenten (92 Blutungen) + 20 Wiederholungen = 215 Ansichten; 2 x 92 / (2 x 92 + 54 + 47) = 0.646; Neubau bitgleich; Urteil steht aus (`ARBEIT.md` 5b) |

Gespeicherte Ergebnisdateien zu diesen Zahlen: `ergebnisse/analysis/` -- `blob_vs_basis_valdo.json` (V8), `operating_point_valdo_members.json` (V9, alle
Einzelmodelle), `operating_point_valdo_ensembles.json` und `ensemble2_valdo.json` (V10), `tta_valdo.json`, `operating_point_valdo_tta.json` (V11),
`subject_level_f1_valdo.json` (V12), `fp_location_valdo.json`, `missed_lesions_valdo_blob.json` (V13).

Determinismus: gleicher Startwert -> gleiche Falten, gleiche Stichprobe; nachgewiesen fuer a01 (vier Nachkommastellen). Zwischen Startwerten streut das
Rezept staerker als bis zum 21.09. angenommen: drei Startwerte des identischen Rezepts 0.646 / 0.590 / 0.578 mit Liquor-Regel (roh 0.585 / 0.553 / 0.501), und Startwert 2 gegen 42 ist im
Fall-Bootstrap "gesichert" schlechter (-0.083 [-0.135; -0.033]). Der Bootstrap ueber Faelle enthaelt das Trainingsrauschen NICHT. Regel seit 22.09.: ein Effekt aus einem Startwert je Seite
gilt erst ab ~0.08 als belastbar; kleinere brauchen einen zweiten Startwert oder eine unabhaengige erste Stufe (`ARBEIT.md` 3, 5u).

---

## 3 catalina -- Schritt fuer Schritt (alles unter `/home/uchralt/data/work/mb-arena`, privat)

| Nr. | Schritt | Befehl | Ergebnis / Pruefzahl |
|---|---|---|---|
| C1 | Hirnmaske + Biaskorrektur | `python hirn_und_bias.py t2star --alle` bzw. `swi` (FSL `fast -B` fuer alle; HD-BET aus Umgebung `medizin` nur fuer die 17 T2*-Faelle ohne mitgelieferte Maske -- 58 T2*- und 61 SWI-Masken stammen aus der Quelle) | `~/data/bids/catalina-<mod>/derivatives/{brainmask,biascorr}/` |
| C2 | Kohorte und Falten | `python kohorte_bauen.py` (Seed 0; Klassen nach CMB-Zahl 0 / 1-2 / 3-5 / > 5, je Klasse reihum auf 5 Falten) | `faelle.json`, `participants.tsv`: T2* 75 Faelle, SWI 61; Falten NICHT mehr aendern |
| C3 | Trainingsgitter | `$PY -m cmb.transfer.build_private_grids --modality t2star` (und `swi`) -- dieselben Funktionen wie V4 | `gitter/<mod>/<id>/...`, `gitter/manifest_<mod>.json` (Kennungen `c2s-` / `csw-`); 828 bzw. 652 Laesionen auf dem Gitter |
| C4 | Gewebekarte T2* | SynthSeg 2.0 `--robust` direkt auf dem T2*-Bild, CPU (Befehl und Begruendung: privates Protokoll 19.09.; ohne `--robust` landen 24 % der CMB im Liquor) | `synthseg-t2star/robust/<id>_synthseg.nii.gz`, 75/75, 1 mm isotrop -- die Werkzeuge rechnen sie selbst auf das Gitter um |
| C5 | eigenes Training | `$PY code/turnier.py --netz a03-aniso --epochen 60 --manifest <mb-arena>/gitter/manifest_t2star.json --name cat-t2s --basis <mb-arena>/transfer` (SWI: `manifest_swi.json`, `--name cat-swi`) | `transfer/a03-aniso-e60-cat-t2s/`, `.../a03-aniso-e60-cat-swi/` |
| C6 | Feintuning | wie C5 mit `--epochen 30 --lr 1e-4 --init <valdo-t2s>/ergebnisse/turnier/a03-aniso-e60-gd/fold{fold}/modell.pt --name ft-t2s` | `transfer/a03-aniso-e30-lr0.0001-ft-t2s/` |
| C7 | Zero-shot | `$PY -m cmb.transfer.zero_shot --models ergebnisse/turnier/a03-aniso-e60-gd --net a03-aniso --manifest ... --out <mb-arena>/transfer/zero-shot-t2star` | Mittel der fuenf VALDO-Faltenmodelle |
| C8 | gemischt | wie C5 mit `--manifest dev/manifest_valdo_gitter_d.json <mb-arena>/gitter/manifest_t2star.json --name mix-valdo-t2s` (alle drei: zusaetzlich `manifest_swi_mix.json`, `--name mix-alle`) | gemeinsame 5 Falten ueber alle Kohorten |
| C9 | Auswertung | `$PY -m cmb.transfer.evaluate --run nur-catalina=... feintuning=... zero-shot=... gemischt=... --manifest ...` (erster Lauf = Bezug der gepaarten Vergleiche) | siehe Tabelle unten |
| C10 | Liquor-Regel, Fehlerbild, Pruefseite | `cmb.analysis.operating_point` / `missed_lesions` / `fp_location` mit `--synthseg "<mb-arena>/synthseg-t2star/robust/{sid}_synthseg.nii.gz"`; `cmb.review.build_qc --out <mb-arena>/qc/...` | 0.675 mit Liquor-Regel; `qc/t2star-nur-catalina/qc.html` |

Ergebnisse T2* (feste Nachbearbeitung, 75 Faelle, gepaart gegen eigenes Training; `transfer/auswertung_t2star_alle_0920.json` u. a.):

| Strategie | F1 | Precision | Sensitivitaet | Fehlalarme je Fall | gepaart |
|---|---|---|---|---|---|
| SHIVA-CMB unveraendert | 0.312 | 0.92 | 0.19 | 0.2 | -- |
| Zero-shot (nur VALDO) | 0.622 | 0.60 | 0.65 | 4.8 | -0.041 [-0.069; -0.016] |
| gemischt VALDO + catalina, 60 Ep. | 0.623 | 0.56 | 0.70 | 6.1 | -0.040 [-0.077; -0.012] |
| alle drei Kohorten gemischt, 60 Ep. | 0.618 | 0.58 | 0.66 | 5.3 | -0.044 [-0.082; -0.018] |
| Feintuning 10 Ep. | 0.619 | 0.56 | 0.69 | 6.0 | -0.044 [-0.090; -0.003] |
| **Feintuning 30 Ep.** | 0.657 | 0.63 | 0.69 | 4.4 | -0.006 [-0.029; +0.015] |
| **eigenes Training 60 Ep.** | 0.663 | 0.67 | 0.66 | 3.6 | Bezug |
| eigenes Training + Feintuning gemittelt | 0.669 | 0.66 | 0.68 | 3.9 | +0.007 [-0.007; +0.021] |
| laesionsweiser Verlust | 0.634 | 0.58 | 0.70 | 5.7 | -0.029 [-0.059; -0.007] |
| Spiegel-TTA | 0.663 | 0.67 | 0.65 | 3.5 | +0.000 [-0.012; +0.014] |

SWI (61 Faelle; `transfer/auswertung_swi_alle_0920.json`, `auswertung_swi_mixalle_0920.json`): eigenes Training **0.634**; Feintuning vom T2*-Netz 0.599
(-0.036 [-0.074; +0.003]); alle Kohorten gemischt 0.547 (-0.087 [-0.126; -0.048]); Zero-shot 0.305 (-0.330 [-0.441; -0.220]).

Warnung zur Lesart (5i): drei T2*-Faelle tragen 234 der 828 Blutungen und 145 der 283 Uebersehenen. Jede Aufschluesselung gepoolter Zahlen braucht die
Gegenprobe ohne diese Faelle (`missed_lesions.py`, Tabelle 5); ohne sie: Sensitivitaet 0.77, F1 0.696. Ob die vielen 1-2-Voxel-Markierungen im Kleinhirn
dieser Faelle sichere Blutungen sind, ist eine offene Expertenfrage.

---

## 3a Zweite Stufe: ein Klassifikator fuer alle Kohorten (`MB` = `/home/uchralt/data/work/mb-arena`; Patches sind Bilddaten -> private Kohorten nur unter `$MB`)

| Nr. | Schritt | Befehl | Ergebnis / Pruefzahl |
|---|---|---|---|
| S1 | Kandidaten VALDO (weich = ohne harte Liquor-Regel, Gewebeklasse als Feld) | `$PY -m cmb.stage2.candidates --predictions ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d --keep-csf --out ergebnisse/stage2/a03-aniso-e60-gd-weich` | 278 Kandidaten (106 auf einer Blutung), erreichen 103 von 139 Blutungen (Obergrenze Sensitivitaet 0.74) |
| S2 | Kandidaten catalina-T2* | `$PY -m cmb.stage2.candidates --predictions $MB/transfer/a03-aniso-e60-cat-t2s --manifest $MB/gitter/manifest_t2star.json --synthseg "$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" --keep-csf --out $MB/stage2/a03-aniso-e60-cat-t2s-weich` | 1082 Kandidaten |
| S3 | Kandidaten catalina-SWI (Startwert 42 und, zur Bestaetigung, Startwert 1) | wie S2 mit `--predictions $MB/transfer/a03-aniso-e60-cat-swi` (bzw. `a03-aniso-s1-e60-cat-swi`), `manifest_swi.json`, `synthseg-swi/robust`, `--out $MB/stage2/a03-aniso-e60-cat-swi-weich` (bzw. `a03-aniso-s1-e60-cat-swi-weich`) | 870 Kandidaten (464 auf einer Blutung), erreichen 461 von 652 (0.71) |
| S4 | gemeinsames 3D-CNN, geschachtelte 5-fach-CV (GPU ~25 min) | `$PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich swi=$MB/stage2/a03-aniso-e60-cat-swi-weich --folds swi=$MB/gitter/manifest_swi_mix.json --models cnn --skip-own --tissue --json $MB/stage2/pooled_weich_cnn_0921.json` (`--folds`: sieben SWI-Faelle sind Zwillinge von T2*-Probanden und muessen in deren Falte) | VALDO 0.625 / T2* 0.672 / SWI 0.670; gepaart gegen die reine Schwelle +0.059 [-0.002; +0.113] / +0.017 [+0.002; +0.041] / **+0.044 [+0.012; +0.085]**; gegen Schwelle + harte Liquor-Regel -0.003 / +0.001 / +0.056 [+0.011; +0.103] |
| S5 | Bestaetigung mit unabhaengiger erster SWI-Stufe | wie S4 mit `swi=$MB/stage2/a03-aniso-s1-e60-cat-swi-weich`, `--json $MB/stage2/pooled_weich_cnn_swi_s1_0921.json` | SWI 0.662 gegen 0.617: **+0.045 [+0.018; +0.085]**; VALDO 0.636, T2* 0.672 |
| S6 | mehr Kontext (eingereiht 21.09., `ARBEIT.md` 5r) | S1-S3 zusaetzlich mit `--half 32 32 16 --out ...-weich-h32`; dann S4 mit den `-h32`-Ordnern und `--models cnn cnn2s` | Kandidaten-Zeilen identisch zu S1-S3, die Mitte der grossen Ausschnitte ist voxelgleich mit den kleinen (geprueft fuer alle vier Saetze); `$MB/stage2/pooled_h32_cnn_cnn2s_0921.json`: gepaart zweiskalig - fein +0.003 / +0.016 [-0.002; +0.030] / +0.016 [-0.002; +0.037]; Bestaetigung `..._swi_s1_0921.json`: **-0.030 [-0.059; -0.011]** / +0.019 [+0.001; +0.036] / +0.013 [-0.001; +0.031] -> nicht uebernommen (Schritt 24); paarweiser Vergleich: `$PY -m cmb.stage2.compare_runs --a <json> "pooled cnn2s" --b <json> "pooled cnn"` |

Die Befehle S1-S3 standen bis zum 21.09. abends nirgends woertlich; sie wurden aus den Optionen rekonstruiert und fuer ALLE VIER Kandidatensaetze GEPRUEFT: der Neubau (mit `--half 32 32 16`) liefert dieselben Zeilen (278 / 1082 / 870 / 893) und in der Mitte der Ausschnitte dieselben Voxel. S4 / S5 stammen woertlich aus der
Schlangen-Datei (Eintraege 137 und 144). Die GPU rechnet nicht bitgleich: zwei Laeufe desselben `pooled cnn` lagen auf VALDO 0.005 auseinander (0.625 / 0.630).

---

## 3b Einmalige Messung auf den 15 beiseitegelegten Faellen (28.09.2026; `ergebnisse/validierung/heldout_0928/`)

| Nr. | Schritt | Befehl | Ergebnis / Pruefzahl |
|---|---|---|---|
| H1 | Manifest der 15 Faelle (Hilfs-Falte 0) | `$PY -m cmb.transfer.build_valdo_manifest --grid dev/gitter_d --held-out --out dev/manifest_valdo_gitter_d_heldout.json` | 15 Faelle, 97 CMB |
| H2 | Erste Stufe je Startwert (GPU) | `$PY -m cmb.transfer.zero_shot --models ergebnisse/turnier/a03-aniso-e60-gd --net a03-aniso --manifest dev/manifest_valdo_gitter_d_heldout.json --out ergebnisse/validierung/heldout_0928/stufe1-s42` (und `a03-aniso-s1-e60-gd` -> `stufe1-s1`) | 2 x 15 Karten, 377 s |
| H3 | Zwei-Startwerte-Ensemble | `$PY -m cmb.analysis.ensemble --runs .../stufe1-s42 .../stufe1-s1 --out .../stufe1-ens2` | 15 Faelle |
| H4 | Auswertung | `cmb.transfer.evaluate` (roh), `cmb.analysis.operating_point --synthseg "dev/synthseg/{id}_synthseg.nii.gz"` (feste Zelle), `cmb.analysis.subject_level_f1 --synthseg ...`, je mit dem Manifest H1 bzw. `..._ohne110.json` / `..._nur110.json` | gepoolt 0.468 (Liquor-Regel) / roh 0.463; ohne sub-110 0.731; Median je Fall 0.667 |

## 3c Forschungslinie 5ae: Synthese und kubisches Gitter (28.09.2026)

| Nr. | Schritt | Befehl | Ergebnis / Pruefzahl |
|---|---|---|---|
| F1 | Verteilung der echten Blutungen | `$PY -m cmb.synthesis.lesion_stats --manifest dev/manifest_valdo_gitter_d.json --json ergebnisse/analysis/lesion_stats_valdo_0928.json` | 139 Laesionen, Volumen-Median 10 mm3, Spitzenkontrast-Median 0.201 |
| F2 | Synthesegitter | `$PY -m cmb.synthesis.build_synthetic_grid --manifest dev/manifest_valdo_gitter_d.json --stats ergebnisse/analysis/lesion_stats_valdo_0928.json --out dev/gitter_s --proc 8 --seed 0` | 57 Faelle, 209 kuenstliche CMB, 319 Verwechsler, checks 57/57 (`dev/gitter_s/summary.json`) |
| F3 | Kubisches Gitter | `GITTER_C_QUELLE=$ROOT/dev/preproc_roh GITTER_C_ZIEL=$ROOT/dev/gitter_k GITTER_C_ORDNUNG=3 $PY code/gitter_c_bauen.py --faelle <57 CV-Ids> --proc 8` (linear vorher an sub-201 bitgleich zu gitter_d geprueft) | 57 Faelle, Label/Maske bitgleich zu gitter_d |
| F4 | Trainings (Schlange 203-206) | `$PY code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_k [--seed 1]` und `... --gitter dev/gitter_d --synthese dev/gitter_s [--seed 1]` | Ordner `a03-aniso-e60-gk`, `a03-aniso-s1-e60-gk`, `a03-aniso-e60-gd-syn`, `a03-aniso-s1-e60-gd-syn` |
| F5 | Momeni/CSIRO-Gitter (Arm 3) | `$PY -m cmb.transfer.build_momeni_grid --processes 8` (Quelle `~/data/extern/momeni-csiro-cmb`, Orte aus `rCMB_locations.json`) | 57 Faelle, 146 Laesionen, Manifest `<extern>/gitter/manifest_momeni.json` |
| F6 | Zero-shot auf Momeni (Schlange 207-209) | `$PY -m cmb.transfer.zero_shot --models $MB/transfer/a03-aniso-e60-cat-swi --net a03-aniso --manifest <extern>/gitter/manifest_momeni.json --out <extern>/transfer/zero-shot-swi-s42` (ebenso `a03-aniso-s1-e60-cat-swi`, `ergebnisse/turnier/a03-aniso-e60-gd`) | Auswertung `cmb.analysis.subject_level_f1` + `cmb.transfer.evaluate` |
| F7 | Arm 4: Klassifikator VALDO -> catalina (Schlange 210-212) | `$PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich swi=$MB/stage2/a03-aniso-e60-cat-swi-weich --folds swi=$MB/gitter/manifest_swi_mix.json --models cnn --skip-own --tissue --transfer valdo=t2star,swi --fractions 0 0.1 0.25 0.5 1 --seed-offset N --json $MB/stage2/transfer_valdo_N.json` (N = 0, 1, 2) | Rauchtest 1 Epoche CPU ok (28.09. 19:20); Modi `transfer<Anteil> cnn` gegen `pooled cnn` im selben Lauf |
| F8 | Arm 1, zweite Fassung 5ae-1b (Schlange 213-214) | Gitter: `$PY -m cmb.synthesis.build_synthetic_grid --manifest dev/manifest_valdo_gitter_d.json --stats ergebnisse/analysis/lesion_stats_valdo_0928.json --out dev/gitter_s2 --proc 8 --seed 1 --profile flat --m-min 0 --m-max 0`; Training: `$PY code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d --synthese dev/gitter_s2 --synthese-anteil 0.5 [--seed 1]` | 57 Faelle, checks 57/57, keine Verwechsler; Ordner `a03-aniso-e60-gd-syn50-s2`, `a03-aniso-s1-e60-gd-syn50-s2`; Wrapper-Test: 62 echte Laesionsausschnitte bitgleich, 19 von 38 leeren ersetzt |
| F9 | Anwendungsmodus: VALDO-Klassifikator auf fremde Kandidaten (29.09.) | Kandidaten: `$PY -m cmb.stage2.candidates --predictions <extern>/transfer/zero-shot-valdo-s42 --manifest <extern>/gitter/manifest_momeni.json --synthseg "<extern>/synthseg/{id}_synthseg.nii.gz" --keep-csf --out <extern>/stage2/zero-shot-valdo-s42-weich` (ebenso swi-s42); Anwendung: `$PY -m cmb.stage2.apply --source ergebnisse/stage2/a03-aniso-e60-gd-weich --target momeni-valdo-unet=<extern>/stage2/zero-shot-valdo-s42-weich momeni-swi-unet=<extern>/stage2/zero-shot-swi-s42-weich --seeds 3 --json ergebnisse/analysis/5ae_0929/momeni_apply_valdo_classifier.json` (GPU, ~12 min) | Schwelle 0.50 aus VALDO; Momeni F1 0.607 (VALDO-U-Net) / 0.630 (SWI-U-Net) |
| F10 | Arm 5: Laufstrecke/Frangi je Kandidat (CPU) und Klassifikator mit 32 mm z-Kontext (Schlange 215-217) | `$PY -m cmb.analysis.vessel_run_length --run NAME=FOLDER --manifest M --synthseg PATTERN --json out.json --table tabelle.json`; Kandidaten `cmb.stage2.candidates ... --keep-csf --half 20 20 20 --out ...-weich-z40` (Zeilen identisch zu `-weich`, geprueft); `$PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40 swi=$MB/stage2/a03-aniso-e60-cat-swi-weich-z40 --folds swi=$MB/gitter/manifest_swi_mix.json --models cnn cnn-z32 --skip-own --tissue --seed-offset N --json $MB/stage2/z32_N.json` | Laufstrecke nur Momeni +0.044; Frangi nirgends; z32 offen |

## 4 Auswertungsregeln (gelten fuer jede Zahl in diesem Dokument)

- Rangmass: gepooltes Laesions-F1, 26er-Komponenten, Beruehrungsregel (`code/metric.py::hits`). Immer out-of-fold.
- EINE feste Nachbearbeitung fuer alle Laeufe und Kohorten (0.3 / 2 mm3 / > 2100 Voxel weg), am 19.09. auf VALDO kreuzweise gewaehlt und seither nicht
  mehr je Lauf angepasst. Die kreuzweise Neuwahl je Modell (`operating_point.py`) wird nur zur Information berichtet -- sie schwankt bei 57 Faellen um
  0.02-0.05 und taugt nicht als Entscheidungsgrundlage.
- Vergleiche immer gepaart ueber dieselben Faelle, Bootstrap ueber Faelle, 10 000 Ziehungen, 95-%-Intervall (`evaluate.py`, erster `--run` = Bezug).
- **Der Fall-Bootstrap enthaelt das Trainingsrauschen nicht** (22.09., Schritt 29): zwei Laeufe des identischen Rezepts mit anderem Startwert liegen bis zu 0.08 auseinander, und das Intervall weist das
  als "gesichert" aus. Deshalb: ein Effekt aus einem Startwert je Seite gilt erst ab ~0.08 als belastbar; kleinere Effekte brauchen einen zweiten Startwert (Training) bzw. eine unabhaengige erste Stufe
  (zweite Stufe, dort ~0.03 Trainingsrauschen bei identischen Kandidaten, Schritt 24). Einzelbefunde unter dieser Schwelle stehen in Abschnitt 5 als Richtungsangaben, nicht als Beweise.
- Liquor-Regel: Komponente faellt weg, wenn ihr SynthSeg-Mehrheitslabel in {4, 5, 14, 15, 24, 43, 44} liegt; parameterfrei.
- Zusaetzlich fuer die Einordnung: VALDO-Regel (`code/metric_valdo.py`: Schwerpunkt < 5 mm, eins zu eins, F1 je Proband) ueber `subject_level_f1.py`.

---

## 5 Der Weg: jede Entscheidung mit Beleg

| Nr. | Schritt | Wirkung (VALDO-CV, gepooltes F1) | Entscheidung | Beleg |
|---|---|---|---|---|
| 1 | Turnier mit zehn U-Net-Bauarten, 40 Ep. | Attention-U-Net 0.245; anisotropes U-Net **0.404** (bestes) | a03-aniso | `ARBEIT.md` 4.4; `code/turnier_rangliste.py` |
| 2 | Fruehstopp -> fester 60-Epochen-Plan (Zwischenstand weiter ueber die innere Falte gewaehlt) | 0.404 -> 0.470; Startwert-Rauschen sinkt | 60-Ep.-Plan | 4.5, Abschnitt 3, 11 |
| 3 | Original-Gefaess-Uebermalung weglassen | 0.470 -> **0.588**, gepaart +0.136 [+0.065; +0.226]; sie traf 109 von 236 CMB | gestrichen | 4.1 |
| 4 | eigene, schonende Uebermalung in drei Dosen | 0.588 / 0.584 / 0.570 -- nirgends Gewinn | nicht uebernommen | 4.1 |
| 5 | Liquor-Regel (SynthSeg) | 0.588 -> 0.612 | uebernommen | 4.3 |
| 6 | Schwelle 0.3 + Mindestgroesse 2 mm3, kreuzweise | 0.633 (am 20.09. mit groeberem Gitter nachgerechnet: feste Zelle 0.646, kreuzweise 0.622) | feste Zelle | 5, 5j |
| 7 | Biaskorrektur / FRST-Kanal / z = 1 mm je einzeln weggelassen | -0.022 / -0.053 / -0.036 | alle drei bleiben | 4.9, 4.11, 4.8 |
| 8 | z-Normierung + Dreh-Augmentierung; enge SynthSeg-Eingangsmaske | -0.060; -0.048 | verworfen | 4.7, 4.2 |
| 9 | zweite Stufe (Schwelle, logistisch, Wald, Boosting, kleines 3D-CNN) | hoechstens gleichauf mit reiner Schwelle -- auch mit 961-1135 catalina-Kandidaten | verworfen | 5d, 5j |
| 10 | SHIVA-CMB als fremde Messlatte | VALDO 0.575 (kennt 60 der 72 Faelle), catalina 0.312 | nur Vergleich | 5a |
| 11 | Transfer auf catalina: vier Strategien, dann Epochen | eigenes Training = Feintuning 30 Ep.; 10 Ep. zu kurz; Mischen und Zero-shot gesichert schlechter | Tabelle in 3 | 5e, 5h |
| 12 | Orts- und Fehleranalyse | kein schwacher Ort; uebersehen wird, was klein UND kontrastarm ist; drei Extremfaelle | Lesewarnung | 5i |
| 13 | laesionsweiser Verlust (blob loss) | VALDO: kleine CMB 0.52 -> 0.64, F1 -0.044 [-0.105; +0.003], zweiter Startwert -0.063 [-0.129; +0.006]; catalina-T2* -0.029 gesichert | nicht uebernommen | 5j, 5m |
| 14 | Zentrumskarte als Hilfsausgabe (nach CenSynCMB) | kleine CMB 0.52 -> 0.66, F1 -0.054 [-0.110; -0.006] | nicht uebernommen | 5k |
| 15 | Ensemble ueber Startwerte; Spiegel-TTA | Ensemble ~ bestes Mitglied, +0.02-0.04 ueber dem mittleren, flach ab zwei; TTA VALDO +0.02, catalina +-0 | Ensemble aus zwei als Standard fuer VALDO | 5j |
| 15a | Verhaeltnis positiv:negativ 1:2 (33 % Laesions-Ausschnitte) | Training kollabiert in die leere Vorhersage (2 von 2 Falten; Test-F1 0.09), Lauf abgebrochen | 60 % bleiben; 1:3 ... entfaellt | 5g |
| 16 | harte Negative (Online-Schuerfen eigener Fehlalarme) | +0.000 [-0.044; +0.048]; wirkt nur auf den Trainingsfaellen | nicht uebernommen | 5l |
| 17 | Neugewichtung nach Uebersehenen (fnrw, nach CenSynCMB) | -0.030 [-0.080; +0.019]; empfindlicher, mehr Fehlalarme | nicht uebernommen | 5l |
| 18 | laesionsweiser Verlust auf catalina-SWI | -0.003 [-0.036; +0.024]; kleine CMB 0.37 -> 0.44 | nicht uebernommen | 5l |
| 19 | Gewebekarte (SynthSeg) als fuenf EINGABEkanaele des U-Net | VALDO -0.058 [-0.138; +0.013]; das Netz lernt die Liquor-Regel selbst (0 Vorhersagen im Liquor), macht aber sonst mehr Fehlalarme | nicht uebernommen (catalina-Lauf steht aus) | 5m |
| 20 | EIN Klassifikator fuer alle Kohorten, Gewebeklasse als Merkmal, Kandidaten empfindlicher Netze als Zusatztraining (`cmb/stage2/pooled.py --tissue --extra ...`) | gegen die beste einfache Kette je Kohorte: VALDO -0.009, T2* +0.011, **SWI +0.033 [+0.013; +0.057]** (0.626 -> 0.659; Fehlalarme je Fall 3.5 -> 2.9) | Kandidat fuer die Endkette; Bestaetigung mit zweitem SWI-Startwert eingereiht; 3D-CNN siehe Schritt 22 | 5j |
| 21 | erst auf allen Kohorten vortrainieren, dann auf die Zielkohorte feintunen | T2* -0.039 [-0.083; -0.007], SWI -0.045 [-0.073; -0.021] gegen eigenes Training | verworfen | 5h |
| 20a | Bestaetigungsversuch zu Schritt 20 mit unabhaengiger SWI-Stufe (Startwert 1) | Wald + Gewebemerkmal + Zusatz: +0.004 [-0.021; +0.041] gegen reine Schwelle -- der SWI-Gewinn der Merkmals-Klassifikatoren ist NICHT bestaetigt | zweite Stufe vorerst nicht Teil der Endkette; 3D-CNN-Bestaetigung eingereiht | 5j |
| 20b | Bestaetigung des gemeinsamen 3D-CNN mit derselben unabhaengigen SWI-Stufe | SWI 0.662 gegen 0.617, gepaart **+0.045 [+0.018; +0.085]** (mit Startwert 42: +0.044) -- reproduziert; VALDO +0.009, T2* +0.001 gegen die harte-Regel-Kette | 3D-CNN wird Teil der Endkette | 5j |
| 22 | gemeinsames 3D-CNN (frueher faelschlich "3D-ResNet" genannt) als zweite Stufe (`pooled.py --models cnn`, mit Zusatzkandidaten) | VALDO 0.622 (gleichauf), T2* 0.684 (+0.014 n. s.), **SWI 0.669 (+0.044 [+0.013; +0.084])** | Kandidat fuer die Endkette | 5j |
| 23 | Gegenprobe: fester LETZTER Zwischenstand statt Wahl auf der inneren Falte (`turnier.py --letzter-stand`, Startwerte 42 und 1) | Abstand der Startwerte schrumpft (0.032 -> 0.007 roh; 0.056 -> 0.020 mit Liquor-Regel), aber das Mittel sinkt (0.569 -> 0.542; 0.618 -> 0.604); s42 gepaart -0.047 [-0.088; -0.010], s1 -0.007 [-0.068; +0.051]; Zwei-Startwerte-Ensemble 0.626 gegen 0.647 | Zwischenstandswahl bleibt; berichtet wird das Mittel ueber Startwerte, nicht der beste Lauf | 5o |
| 24 | zweite Stufe mit mehr Kontext: zweiskaliges CNN (16 mm fein + 28 mm grob) gegen das feine CNN auf denselben Kandidaten, zwei Laeufe (zweiter mit unabhaengiger SWI-Stufe) | VALDO +0.003 / **-0.030 [-0.059; -0.011]**, T2* +0.016 / **+0.019 [+0.001; +0.036]**, SWI +0.016 / +0.013 (n. s.); auf T2* sinken die Fehlalarme bei gleicher Sensitivitaet; VALDO schwankt bei identischen Kandidaten um 0.03 zwischen den Laeufen = Trainingsrauschen der zweiten Stufe | nicht uebernommen; feines CNN bleibt | 5r |
| 25 | Budget-Frage zum Mischen: eigenes catalina-T2*-Training mit 120 Ep. als Kontrolle zu gemischt 120 Ep. | eigenes 120 Ep. 0.640 gegen eigenes 60 Ep. 0.663: **-0.022 [-0.052; -0.002]** (Fehlalarme 3.6 -> 5.0); gemischt 120 gegen eigenes 120: +0.001 [-0.021; +0.023] (vorab festgelegter Vergleich); gemischt 120 gegen eigenes 60: -0.021 [-0.054; +0.006] | Mischen kostet bei gleichem Budget je Fall nichts mehr, bringt aber auch nichts; beste Strategie bleibt eigenes Training 60 Ep.; Budget je Fall (~1000-1400 Ausschnitte) ist die richtige Groesse, nicht die Epochenzahl | 5h |
| 26 | "120 Epochen generell?": Basisrezept auf VALDO mit 120 Ep. (Vorhersage vorab: kein Gewinn) | 0.556 gegen 0.585 roh (-0.029 [-0.068; +0.019]), 0.610 gegen 0.646 mit Liquor-Regel; Fehlalarme 1.51 -> 1.84; Kohorte 0.8 mm gesichert schlechter (-0.046) | nein -- die richtige Groesse ist Ausschnitte je Trainingsfall (~1000-1400); 120 Ep. nur bei entsprechend mehr Faellen | 5h |
| 27 | Ausschnittgroesse, klein (24 x 48 x 48 Voxel = 24 mm-Wuerfel, nahe an Sundaresans 48^3) | 0.369 gegen 0.585 roh, **-0.216 [-0.277; -0.156]**; mit Liquor-Regel 0.491 gegen 0.646; S 0.74, aber 5.5 Fehlalarme je Fall; alle drei Kohorten negativ | verworfen -- fuer ein einstufiges Netz ist 24 mm Kontext zu wenig; grosser Ausschnitt 64 mm laeuft | 5t |
| 28 | Ausschnittgroesse, gross (48 x 96 x 96 Voxel = 48-mm-Wuerfel, 2x Voxel des Rezepts; 64 mm nicht messbar: Speichergrenze der cgroup) | 0.569 gegen 0.585 roh (-0.016 [-0.074; +0.034]), 0.609 gegen 0.646 mit Liquor-Regel; je Kohorte alle n. s.; 29 statt 21 min je Falte | nicht uebernommen; Dosisreihe 24 / Rezept / 48 mm: 0.369 / 0.585 / 0.569 -- das Rezept liegt am Optimum | 5t |
| 29 | dritter Startwert der Basis (`--seed 2`) und Drei-Startwerte-Ensemble | s2 0.501 roh / 0.578 mit Regel; gegen s42 **-0.083 [-0.135; -0.033]** (identisches Rezept!); Ensemble 3: 0.636 gegen Ensemble 2: 0.647 | Rauschmass auf ~0.08 angehoben (Abschnitt 4 hier, `ARBEIT.md` 3); Basis als Mittel 0.605 +- 0.034 berichten; Zwei-Startwerte-Ensemble bleibt Standard | 5u |
| 30 | Dickschicht-Simulation als Augmentierung (`--aug-schicht`; Hypothese: Gewinn in den dicken Kohorten) | 0.458 gegen 0.585 roh, **-0.127 [-0.179; -0.075]** (ueber dem Rauschmass); je Kohorte -0.045 / -0.125 / -0.146 -- der groesste Verlust genau dort, wo der Gewinn vorhergesagt war; 3.3 Fehlalarme je Fall | verworfen; Lehre: Augmentierung muss Bild UND Etikett-Statistik des Ziels treffen; Zero-shot auf catalina laeuft noch (Hauptmass 5p) | 5p |
| 31 | Intensitaets-Augmentierung (`--aug-intensitaet`: Gamma, Bias-Feld, Kontrast, Helligkeit, Rauschen, je p = 0.5; Erwartung: Gleichstand) | 0.477 gegen 0.585 roh, **-0.108 [-0.180; -0.030]**; S 0.56 UND P 0.41 fallen; unter dem schwaechsten Startwert (0.501) | innerhalb VALDO verworfen (zu starke Stoerung fuer das Budget); Zero-shot auf catalina laeuft (Hauptmass 5p); Robustheitsprobe (5v) prueft, ob die Basis schon invariant ist | 5p, 5v |
| 32 | Augmentierung, Hauptmass: Zero-shot auf catalina (Rauschmass = Zero-shot der Basis mit Startwert 1) | T2*: s1 +0.016 [+0.003; +0.033], Intensitaet -0.014, Dickschicht -0.016; SWI: s1 +0.006, Intensitaet -0.007, Dickschicht **-0.065 [-0.096; -0.036]** (16 Fehlalarme je Fall); ueberall MEHR Fehlalarme statt weniger | beide Augmentierungen gestrichen; Nebenbefund: der in der CV schwaechere Startwert uebertraegt besser -> Zwei-Startwerte-Ensemble ausliefern, nie "den besten Startwert" | 5p |
| 33 | Architektur-Gegenprobe auf sauberen Daten, Teil 1: nnU-Net-Bauart `a02-nnunet` im heutigen Rezept (60 Ep., gitter_d) | 0.458 gegen 0.585 roh, **-0.127 [-0.202; -0.048]**; 0.515 gegen 0.646 mit Liquor-Regel; alle fuenf Falten und alle drei Kohorten hinten; S UND P niedriger | Rangfolge des Turniers haelt; Basis bleibt; Attention-U-Net laeuft | 5s |
| 34 | Architektur-Gegenprobe, Teil 2: Attention-U-Net `a01-attunet` im heutigen Rezept | 0.463 gegen 0.585 roh, **-0.121 [-0.190; -0.061]**; 0.536 gegen 0.646 mit Liquor-Regel; S 0.68 wie die Basis, aber P 0.35 und 3.1 Fehlalarme je Fall; Kohorte 3-4 mm **-0.200** | Rangfolge des Turniers auf sauberen Daten bestaetigt (2 von 2 Herausforderern); Basis bleibt endgueltig | 5s |
| 35 | Nutzerfrage: kann die zweite Stufe aus der hohen Sensitivitaet des Attention-U-Net F1 machen? Gemessen wird die OBERGRENZE der Kandidatenmenge (`cmb.stage2.candidates --keep-csf`, CPU) | erreichte Blutungen von 139: Basis 103, attunet **104** (bei 407 statt 278 Kandidaten), blob 110, nnunet 92; F1-Decke bei fehlerfreier zweiter Stufe 0.851 / 0.856 / 0.884 / 0.797 | nein, kein GPU-Lauf; Regel: eine erste Stufe hilft der Kette nur ueber eine hoehere OBERGRENZE, nicht ueber Empfindlichkeit am Arbeitspunkt (Gegentest blob: Decke 0.884, Kette trotzdem 0.594 gegen 0.625) | 5s, 5j |

Laeuft oder eingereiht (Stand 22.09. 20:27; fertig sind Schritte 23-35 -- Trainingsrezept vermessen, Augmentierung dreifach negativ, Architekturwahl auf sauberen Daten bestaetigt): nur noch die Robustheitsprobe am fertigen Netz
(`cmb.analysis.robustness_probe`, ~2 GPU-h, 5v) und die Robustheitsprobe am fertigen Netz (`cmb.analysis.robustness_probe`, ~2 GPU-h; Verfahren und Erwartung vorab in `ARBEIT.md` 5v). Die Zero-shot-Laeufe der
Augmentierung (`cmb.transfer.zero_shot --models ergebnisse/turnier/a03-aniso-e60-gd-augi|-augs|a03-aniso-s1-e60-gd`) sind ausgewertet (Schritt 32). Gestrichen am 21.09. (Eintraege bleiben mit Begruendung in der Schlangen-Datei): Verhaeltnis 1:1 + Kette, Attention auf dem Sieger, Perzentil-Skala Startwert 1, Feintuning 60 Ep. (T2*, SWI), Gewebekanal auf catalina.
Jeder Trainingsbefehl steht mit Zeitstempel und Rueckgabewert in `/home/uchralt/local_agentic_system/system/logs/grossauftrag-schlange.jsonl`. Die Nummern in eckigen Klammern sind die Anzeige-Nummern ZUM ZEITPUNKT DES EINREIHENS; jedes Vorziehen verschiebt sie (am 21.09. 21:52 um 2) -- massgeblich ist der Befehl, nicht die Nummer.

---

## 6 Grenzen der Replizierbarkeit (ehrlich benannt)

1. Bis zum 20.09. gab es keine Versionsverwaltung; der Code-Stand aelterer Laeufe ist nur ueber datierte Sicherungskopien (`*.vor-*`) und `meta.json` je Falte
   rekonstruierbar. Seit dem 20.09. taeglicher Schnappschuss im privaten Repository.
2. Auswertungen vor dem 20.09. liefen teils als Einmal-Skripte in der Sitzung. Stand 21.09.: ALLE liegen als Datei vor und sind gegen die protokollierten Zahlen geprueft:

   | Auswertung | Datei / Befehl | protokolliert | aus der Datei | Urteil |
   |---|---|---|---|---|
   | Schadensbilanz der Original-Uebermalung | `$PY code/uebermalung2.py --pruefen dev/preproc` | 109 / 61 / 27 / 49 / 17 von 236; 12.2 / 11.7 / 17.3 % des Hirns | identisch | exakt |
   | Maskenvolumen, Nicht-Hirn-Anteil | `$PY -m cmb.analysis.mask_volume --grid dev/gitter_d --synthseg "dev/synthseg/{id}_synthseg.nii.gz"` | 1419 / 1770 / 2533 ml (2085-3009); 4 / 16 / 40 % | identisch | exakt |
   | Kohorten-Aufschluesselung (19.09. 07:00) | `$PY -m cmb.analysis.cohort_breakdown --run ergebnisse/turnier/a03-aniso ergebnisse/turnier/a01-attunet ...` | a03-aniso 0.500 / 0.359 / 0.471 (mit Liquor-Regel 0.510 / 0.478 / 0.490) | identisch | exakt |
   | FRST-Trefferquote an der CMB | `$PY -m cmb.analysis.frst_hit_rate --grid dev/gitter_c dev/gitter_d` | 201/236 = 85 % -> 227/236 = 96 % | **194/236 = 82 % -> 226/236 = 96 %** | weicht ab (7 bzw. 1 Blutung); die Datei rechnet wie `gitter_c_bauen.py::frst_guete` und trifft dessen Bauzeit-Werte der 57 CV-Faelle exakt (124/139, 135/139); der Einmal-Code vom 19.09. ist nicht erhalten. Ab jetzt gelten 82 % / 96 %; die Aussage (Uebermalung schwaecht die FRST in den dickschichtigen Kohorten: Ko1 78 -> 99 von 106, Ko3 26 -> 34 von 34) bleibt. |
   | Kontrastverlust durch die Umrechnung | `$PY -m cmb.analysis.resampling_contrast --source dev/preproc_roh --grid dev/gitter_d --manifest dev/manifest_valdo_gitter_d.json` | -23 / -12 / -32 % (Spitze -20 / -11 / -22 %) | **-26 / -15 / -30 %** (Spitze -17 / -13 / -16 %), Volumen -1 / 0 / 0 % | weicht im Betrag ab, gleiches Muster; die Ringdefinition steht jetzt im Docstring (0.5-2.0 mm in der Schicht, im Hirn, ohne andere Blutungen). Ab jetzt gelten die Werte aus der Datei. |

   Dazu seit 20./21.09.: gepaarter Vergleich und Kohortenschalter (`evaluate.py --split-valdo`), Arbeitspunkt (`operating_point.py`), Ort der Fehler (`fp_location.py`), Uebersehene
   (`missed_lesions.py`), Mass je Proband (`subject_level_f1.py`), Fallliste (`build_valdo_manifest.py`), Ensemble, Spiegel-TTA. Aktuelles Modell je Kohorte (feste Zelle, roh):
   Kohorte 1 (4 mm) 0.557, Kohorte 2 (0.8 mm) 0.634, Kohorte 3 (3-4 mm) 0.479 (`ergebnisse/analysis/basis_je_kohorte_valdo.json`).
3. Schritt 6 der Leiter: die am 19.09. berichtete 0.633 stammt aus einer Einmal-Gittersuche mit anderem Raster; das heutige Werkzeug liefert 0.646 (feste
   Zelle) und 0.622 (kreuzweise). Fuer eine Veroeffentlichung gilt die heutige, als Datei vorliegende Rechnung.
4. Die 15 beiseitegelegten VALDO-Faelle wurden am 14.09. einmal benutzt; alle Zahlen hier stammen aus der 5-fach-CV der uebrigen 57. Die eine Schlussmessung
   steht aus und ist offenzulegen.
5. Startwert-Streuung: fuer die Basis liegen zwei echte Startwerte vor (42, 1; ein dritter ist eingereiht), fuer catalina je Strategie einer. Aussagen zu
   catalina-Strategien stuetzen sich auf gepaarte Intervalle ueber Faelle, nicht ueber Startwerte.
6. catalina ist privat: Schritte C1-C10 sind nur auf Alita nachvollziehbar; nach aussen gehen ausschliesslich Summen.
7. Literaturzahlen in `recherche/` sind groesstenteils aus Suchausschnitten und als "unverified" markiert; an der Quelle geprueft ist CenSynCMB (Abstract und
   Volltext-HTML, 20.09.). Die Fehler im Original (A1-A3) sind am Quelltext gelesen, nicht ausgefuehrt.
