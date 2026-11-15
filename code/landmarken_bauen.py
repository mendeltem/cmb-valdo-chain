"""Erzeugt die Referenz-Landmarken fuer `turnier.py --norm landmarken`.

Warum getrennt und als Datei: die Angleichung braucht ein FESTES Ziel, das bei Training und Vorhersage identisch ist --
auch fuer eine fremde Kohorte, die spaeter dazukommt. Eine Datei laesst sich mitliefern und pruefen; eine im Code
ausgerechnete Referenz waere still von den gerade geladenen Faellen abhaengig.

    python code/landmarken_bauen.py --gitter dev/gitter_d --aus dev/landmarken_valdo.json

WAS HIER OFFENGELEGT WERDEN MUSS: die Referenz ist der Median der Fall-Perzentile ueber die angegebenen Faelle. Nimmt
man dafuer die 57 Kreuzvalidierungs-Faelle, fliesst die Intensitaetsverteilung JEDES Falles zu 1/57 in die Skala ein,
mit der er spaeter auch vorhergesagt wird. Es sind reine Intensitaets-Perzentile, keine Etiketten -- der uebliche Weg
in der Literatur und ein sehr kleiner Einfluss, aber ein Einfluss. Wer das ausschliessen will, baut die Referenz je
Falte aus deren Trainingsfaellen (`--faelle` mit der jeweiligen Liste) und uebergibt sie mit `--landmarken`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/code")
from turnier import LANDMARKEN_PERZENTILE, landmarken_bestimmen     # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gitter", required=True, help="Ordner mit <id>/<id>_image.nii.gz")
    p.add_argument("--faelle", nargs="*", help="nur diese Kennungen (Vorgabe: alle im Gitter)")
    p.add_argument("--aus", required=True)
    a = p.parse_args()
    gitter = a.gitter if os.path.isabs(a.gitter) else f"{ROOT}/{a.gitter}"
    ids = a.faelle or sorted(os.path.basename(d) for d in glob.glob(f"{gitter}/*") if os.path.isdir(d))
    bilder = []
    for i in ids:
        pfad = f"{gitter}/{i}/{i}_image.nii.gz"
        if os.path.exists(pfad):
            bilder.append(np.asarray(nib.load(pfad).dataobj).astype(np.float32))
    if not bilder:
        raise SystemExit(f"keine Bilder in {gitter}")
    werte = landmarken_bestimmen(bilder)
    aus = a.aus if os.path.isabs(a.aus) else f"{ROOT}/{a.aus}"
    json.dump(dict(perzentile=list(LANDMARKEN_PERZENTILE), landmarken=[round(float(v), 6) for v in werte],
                   gitter=gitter, n_faelle=len(bilder), faelle=sorted(ids) if a.faelle else "alle"),
              open(aus, "w"), indent=1)
    print(f"{len(bilder)} Faelle -> Landmarken " + "  ".join(f"P{q}={v:.4f}" for q, v in zip(LANDMARKEN_PERZENTILE, werte)))
    print("geschrieben nach", aus)


if __name__ == "__main__":
    main()
