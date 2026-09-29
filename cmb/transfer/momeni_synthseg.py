"""SynthSeg (--robust, as for the private SWI cohort) for the Momeni/CSIRO cases, resampled onto the training grid, so the
CSF rule and the tissue-aware tools can be applied there too (ARBEIT.md 5ae, arm 3; 2026-09-28).

Input: the authors' skull-stripped, bias-corrected SWI (native, NOT the inverted grid image). Output:
<extern>/synthseg/native/<id>_synthseg.nii.gz (SynthSeg output) and <extern>/synthseg/<id>_synthseg.nii.gz (nearest-neighbour
resampled onto <extern>/gitter/<id>/<id>_image.nii.gz -- same convention as dev/synthseg for VALDO).

    python -m cmb.transfer.momeni_synthseg [--threads 6] [--cases id ...]
"""
from __future__ import annotations
import argparse, json, os, subprocess, time
import nibabel as nib, numpy as np
from nibabel.processing import resample_from_to

EXTERN = "/home/uchralt/data/extern/momeni-csiro-cmb"
FS = "/home/uchralt/freesurfer/8.2.0/bin/mri_synthseg"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=6); ap.add_argument("--cases", nargs="*", default=None); a = ap.parse_args()
    env = dict(os.environ, FREESURFER_HOME="/home/uchralt/freesurfer/8.2.0")
    entries = json.load(open(f"{EXTERN}/gitter/manifest_momeni.json"))
    locs = {("mom-" + e["file"].split("_")[0] + "-" + e["file"].split("_")[1]): e["file"] for e in json.load(open(f"{EXTERN}/rCMB_locations.json"))}
    os.makedirs(f"{EXTERN}/synthseg/native", exist_ok=True)
    done = failed = 0
    for e in entries:
        sid = e["id"]
        if a.cases and sid not in a.cases:
            continue
        out = f"{EXTERN}/synthseg/{sid}_synthseg.nii.gz"
        if os.path.exists(out):
            done += 1; continue
        t0 = time.time()
        native = f"{EXTERN}/synthseg/native/{sid}_synthseg.nii.gz"
        if not os.path.exists(native):
            src = f"{EXTERN}/PublicDataShare_2020/rCMB_DefiniteSubject/{locs[sid]}"
            r = subprocess.run([FS, "--i", src, "--o", native, "--cpu", "--threads", str(a.threads), "--robust"], env=env, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(native):
                print(f"{sid}: FAILED rc={r.returncode} {r.stderr[-300:]}", flush=True); failed += 1; continue
        grid = nib.load(f"{e['dir']}/{sid}_image.nii.gz")
        seg = nib.load(native)
        res = resample_from_to(seg, (grid.shape, grid.affine), order=0)
        labels = np.asarray(res.dataobj).astype(np.int16)
        nib.save(nib.Nifti1Image(labels, grid.affine), out)
        brain = np.asarray(grid.dataobj) > 0
        print(f"{sid}: {len(np.unique(labels))} labels, brain voxels with label 0: {100 * float((labels[brain] == 0).mean()):.1f} %, {time.time() - t0:.0f} s", flush=True)
        done += 1
    print(f"done {done}, failed {failed}")


if __name__ == "__main__":
    main()
