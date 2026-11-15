"""Where the code finds its data: one place, set by environment variables.

The research code was written on one machine with absolute paths. For the public paper version every such path
was replaced by one of the names below (see PAPER_VERSION.md). Each name reads an environment variable; the default
is a folder inside the repository, so a fresh clone works without any setting.

| name            | environment variable  | default                       | what lives there |
|-----------------|-----------------------|-------------------------------|------------------|
| ROOT            | CMB_ROOT              | the repository                | dev/ (grids, manifests, folds), ergebnisse/ (results) |
| PRIVATE         | CMB_PRIVATE           | ROOT/private                  | work folder of your own (non-public) cohorts: grids, manifests, candidates |
| VALDO           | CMB_VALDO             | ROOT/data/valdo2021/Task2     | VALDO 2021 Task 2 (sub-XXX/...) |
| MOMENI          | CMB_MOMENI            | ROOT/data/momeni-csiro-cmb    | the public Momeni/CSIRO SWI cohort |
| VEIN_ATLAS      | CMB_VEIN_ATLAS        | ROOT/data/venous-atlas-bernier2018/.../mean_Ved_swi_Thresh.nii.gz | venous atlas (Bernier 2018) |
| FREESURFER_HOME | FREESURFER_HOME       | /usr/local/freesurfer         | FreeSurfer (mri_synthseg, mri_synthmorph) |
| MNI_TEMPLATE    | CMB_MNI_TEMPLATE      | $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz | MNI template |
| PY              | CMB_PYTHON            | the running interpreter       | Python used for sub-processes |
| MODELS          | CMB_MODELS            | ROOT/models                   | downloaded pretrained weights (cmb.pretrained) |
| INHOUSE_LABEL   | CMB_INHOUSE_LABEL     | inhouse                       | cohort label written into the manifests of own cohorts ("<label>-t2star", "<label>-swi") |
| CASES           | CMB_CASES             | PRIVATE/faelle.json           | case list of own cohorts: {"t2star"|"swi": {"faelle": [{id, fold, bild, maske, hirn, ...}]}} |
| TWINS           | CMB_TWINS             | (empty)                       | JSON {swi_id: t2star_id}: patients scanned in both own cohorts (LODO excludes them) |
| ID_PREFIX       | CMB_ID_PREFIX_T2STAR, CMB_ID_PREFIX_SWI | t2s, swi              | case-id prefix of the own cohorts on the model grid |
| QUEUE           | CMB_QUEUE             | (empty = run directly)        | optional GPU job queue command used by cmb.experiments.kern --queue |

Nothing else changed: with the variables set to the values of the original machine the code runs the same commands
on the same files.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dotenv() -> None:
    """Settings of .env in the repository root become defaults (a variable set in the shell wins); relative paths
    resolve against the repository root."""
    f = os.path.join(_REPO, ".env")
    if not os.path.exists(f):
        return
    for line in open(f):
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        k, v = (s.strip().strip('"').strip("'") for s in line.split("=", 1))
        if k in ("CMB_ROOT", "CMB_PRIVATE", "CMB_VALDO", "CMB_MOMENI", "CMB_CASES", "CMB_TWINS", "CMB_MODELS", "CMB_MODEL_CACHE"):
            os.environ.setdefault(k, v if os.path.isabs(v) else os.path.join(_REPO, v))


_dotenv()

ROOT = os.environ.get("CMB_ROOT", _REPO)
PRIVATE = os.environ.get("CMB_PRIVATE", os.path.join(ROOT, "private"))
MB = PRIVATE                                   # the research code called the private work folder MB
VALDO = os.environ.get("CMB_VALDO", os.path.join(ROOT, "data", "valdo2021", "Task2"))
MOMENI = os.environ.get("CMB_MOMENI", os.path.join(ROOT, "data", "momeni-csiro-cmb"))
VEIN_ATLAS = os.environ.get("CMB_VEIN_ATLAS", os.path.join(
    ROOT, "data", "venous-atlas-bernier2018", "Template_MNI_2018_41_subjects", "mean_Ved_swi_Thresh.nii.gz"))
FREESURFER_HOME = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
MNI_TEMPLATE = os.environ.get("CMB_MNI_TEMPLATE", os.path.join(
    os.environ.get("FSLDIR", "/usr/local/fsl"), "data", "standard", "MNI152_T1_1mm_brain.nii.gz"))
PY = os.environ.get("CMB_PYTHON", sys.executable)
PYBIN = os.path.dirname(PY)
MODELS = os.environ.get("CMB_MODELS", os.path.join(ROOT, "models"))
INHOUSE_LABEL = os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
CASES = os.environ.get("CMB_CASES", os.path.join(PRIVATE, "faelle.json"))   # case list of own cohorts
TWINS = os.environ.get("CMB_TWINS", "")   # optional JSON {swi_id: t2star_id} of patients scanned in both own cohorts
QUEUE = os.environ.get("CMB_QUEUE", "")
# case-id prefix of the own cohorts on the model grid ("<prefix>-<id>", unique across cohorts); set CMB_ID_PREFIX_T2STAR /
# CMB_ID_PREFIX_SWI to the prefixes of existing grids and results
ID_PREFIX = {"t2star": os.environ.get("CMB_ID_PREFIX_T2STAR", "t2s"), "swi": os.environ.get("CMB_ID_PREFIX_SWI", "swi")}


def describe() -> str:
    """One line per setting (python -m cmb.paths)."""
    names = ["ROOT", "PRIVATE", "VALDO", "MOMENI", "VEIN_ATLAS", "FREESURFER_HOME", "MNI_TEMPLATE", "PY", "MODELS",
             "INHOUSE_LABEL", "CASES", "TWINS", "ID_PREFIX", "QUEUE"]
    return "\n".join(f"{n:16s} {globals()[n]}" for n in names)


if __name__ == "__main__":
    print(describe())
