"""VALDO T2* preprocessing for CMB detection: bias correction -> microbleednet chain.

Produces the two channels the network is trained on -- the inverted, vessel-overpainted,
brain-masked image and the FRST map -- plus the label, all on the original image grid.

  python preprocess.py --out <dir> [--cases sub-101 sub-201] [--all] [--skip-bias]

Every stage is timed separately and written to <out>/<id>/timing.json, because the cost is
not where one would guess: FSL FAST scales with voxel count (cohort 2 has 50 million), and
inpaint_vessels runs a KMeans per axial slice (cohort 2 has 192 of them).

The chain, deliberately imported rather than reimplemented (microbleednet,
scripts/data_preparation.py::preprocess_subject):
  binary_fill_holes(img>0) brain mask  ->  inpaint_vessels (Frangi + structure-tensor
  linearity, KMeans k=2; round/solid components survive, the rest is painted over with
  the neighbourhood mean)  ->  FRST (fast_radial_symmetry_xfm per slice, radii [2,3])
  ->  invert 1-(img/max) and mask.
Bias correction (FSL FAST -B) is NOT part of that chain and runs before it."""
import os, sys, json, time, argparse, subprocess, glob
import numpy as np, nibabel as nib

VALDO = "/home/uchralt/data/extern/valdo2021/Task2"
MBNET = "/home/uchralt/software/microbleed-detection"
FSLDIR = "/home/uchralt/fsl"

def bias_correct(t2s_path, workdir):
    """FSL FAST -B on the (already skull-stripped) T2*. Returns path to the restored image."""
    env = dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ",
               PATH=f"{FSLDIR}/bin:{os.environ['PATH']}")
    stem = os.path.join(workdir, "fast")
    out = f"{stem}_restore.nii.gz"
    if not os.path.exists(out):
        r = subprocess.run([f"{FSLDIR}/bin/fast", "-t", "2", "-n", "3", "-B", "-b",
                            "--nopve", "-o", stem, t2s_path],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"fast failed rc={r.returncode}: {r.stderr[-400:]}")
    return out

def already_done(sid, outdir):
    """Resume: a case counts as finished only with all three outputs AND its timing."""
    return all(os.path.exists(f"{outdir}/{sid}/{sid}_{w}.nii.gz") for w in ("image", "frst", "label")) \
           and os.path.exists(f"{outdir}/{sid}/timing.json")

def run_case(sid, outdir, skip_bias=False):
    sys.path.insert(0, MBNET)
    from microbleednet.scripts.data_preparation import preprocess_subject

    d = f"{VALDO}/{sid}"
    t2s = f"{d}/{sid}_space-T2S_desc-masked_T2S.nii.gz"
    cmb = f"{d}/{sid}_space-T2S_CMB.nii.gz"
    case_out = os.path.join(outdir, sid); os.makedirs(case_out, exist_ok=True)
    ref = nib.load(t2s); t = {}

    t0 = time.time()
    img_path = t2s if skip_bias else bias_correct(t2s, case_out)
    t["bias_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    image, label, frst = preprocess_subject({"input_path": img_path, "label_path": cmb})
    t["mbnet_s"] = round(time.time() - t0, 1)

    # Grid check: whatever comes back must still sit on the raw image grid, or the label
    # and the prediction would not line up later.
    for name, a in (("image", image), ("frst", frst), ("label", label)):
        if a.shape != ref.shape:
            raise RuntimeError(f"{sid}: {name} {a.shape} left the image grid {ref.shape}")
    for name, a, dt in (("image", image, np.float32), ("frst", frst, np.float32),
                        ("label", label, np.uint8)):
        nib.save(nib.Nifti1Image(np.asarray(a, dtype=dt), ref.affine, ref.header),
                 f"{case_out}/{sid}_{name}.nii.gz")

    t.update(shape=list(ref.shape), zooms=[round(float(z), 3) for z in ref.header.get_zooms()[:3]],
             cohort=sid[4], voxels=int(np.prod(ref.shape)), slices=int(ref.shape[2]),
             label_voxels=int((label > 0).sum()),
             image_min=round(float(image.min()), 4), image_max=round(float(image.max()), 4),
             image_mean_in_brain=round(float(image[image > 0].mean()) if (image > 0).any() else 0.0, 4),
             frst_max=round(float(frst.max()), 4), skip_bias=skip_bias)
    json.dump(t, open(f"{case_out}/timing.json", "w"), indent=1)
    print(f"{sid}  cohort {t['cohort']}  {t['shape']}  bias {t['bias_s']}s  mbnet {t['mbnet_s']}s  "
          f"label {t['label_voxels']} vox  img mean(brain) {t['image_mean_in_brain']}")
    return t

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-bias", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="skip cases whose output is already complete")
    a = ap.parse_args()
    cases = sorted(os.path.basename(x) for x in glob.glob(f"{VALDO}/sub-*")) if a.all else a.cases
    if not cases: sys.exit("give --cases <id ...> or --all")
    os.makedirs(a.out, exist_ok=True)
    ts = []
    for sid in cases:
        if a.skip_done and already_done(sid, a.out):
            print(f"{sid}  already complete, skipped"); continue
        try:
            ts.append(run_case(sid, a.out, a.skip_bias))
        except Exception as e:
            print(f"{sid}: FAILED {e}"); json.dump({"error": str(e)}, open(f"{a.out}/{sid}.error.json", "w"))
    if ts:
        print(f"\n{len(ts)} cases: bias {sum(t['bias_s'] for t in ts):.0f}s total, "
              f"mbnet {sum(t['mbnet_s'] for t in ts):.0f}s total")
