"""Fehlende Hirnmasken mit HD-BET (CPU, env medizin) erzeugen und FSL-Biaskorrektur (fast -B) rechnen.
Regel des Nutzers (04.09.): wenn Hirnmaske fehlt -> hd-bet und fslbias correction.
Ausgabe (Symlink-frei, gerechnet):
  <bids>/derivatives/brainmask/<id>/anat/<id><stamm>_desc-brain_mask.nii.gz   (nur wo Quelle keine hat)
  <bids>/derivatives/biascorr/<id>/anat/<id><stamm>_desc-biascorr_T2starw.nii.gz  (hirnextrahiert + fast -B)
Aufruf: python hirn_und_bias.py <mod> [--alle]   ohne --alle nur Faelle ohne Quell-Hirnmaske."""
import os, sys, glob, subprocess, shutil, tempfile
import numpy as np, nibabel as nib, pandas as pd
Q = os.path.expanduser("~/data/sourcedata/catalina"); ZIEL = os.path.expanduser("~/data/bids")
MOD = {"swi": ("BIDS_SWI", "swi", "SWI", "_acq-swi_T2starw"), "t2star": ("BIDS_T2_STAR", "t2", "T2", "_T2starw")}
HDBET = os.path.expanduser("~/miniconda3/envs/medizin/bin/hd-bet"); FAST = os.path.expanduser("~/fsl/share/fsl/bin/fast")
FSLENV = dict(os.environ, FSLDIR=os.path.expanduser("~/fsl"), FSLOUTPUTTYPE="NIFTI_GZ")
mod = sys.argv[1]; alle = "--alle" in sys.argv
ordner, unter, suffix, stamm = MOD[mod]; basis = f"{Q}/{ordner}"; ds = f"{ZIEL}/catalina-{mod}"
tsv = pd.read_csv(f"{basis}/participants.tsv", sep="\t")
for sid in tsv.participant_id:
    bild = f"{basis}/{sid}/{unter}/{sid}_{suffix}.nii.gz"; hirn_q = f"{basis}/{sid}/{unter}/{sid}_{suffix}_brainmask.nii.gz"
    if os.path.exists(hirn_q) and not alle: continue
    hirn_z = f"{ds}/derivatives/brainmask/{sid}/anat/{sid}{stamm}_desc-brain_mask.nii.gz"
    bias_z = f"{ds}/derivatives/biascorr/{sid}/anat/{sid}{stamm}_desc-biascorr_T2starw.nii.gz"
    if os.path.exists(bias_z) and (os.path.exists(hirn_z) or os.path.exists(hirn_q)): continue
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/data/work/mb-arena/tmp")) as t:
        if os.path.exists(hirn_q): hirn = hirn_q
        else:
            subprocess.run([HDBET, "-i", bild, "-o", f"{t}/hd.nii.gz", "-device", "cpu", "--disable_tta", "--save_bet_mask"], check=True, capture_output=True)
            m = f"{t}/hd_bet.nii.gz"
            os.makedirs(os.path.dirname(hirn_z), exist_ok=True); shutil.copy(m, hirn_z); hirn = hirn_z
        ib = nib.load(bild); ih = nib.load(hirn); assert ib.shape[:3] == ih.shape[:3], sid
        brain = np.asarray(ib.dataobj).astype(np.float32) * (np.asarray(ih.dataobj) > 0)
        nib.save(nib.Nifti1Image(brain, ib.affine, ib.header), f"{t}/brain.nii.gz")
        subprocess.run([FAST, "-B", "--nopve", "-t", "2", "-o", f"{t}/fast", f"{t}/brain.nii.gz"], check=True, capture_output=True, env=FSLENV)
        os.makedirs(os.path.dirname(bias_z), exist_ok=True); shutil.copy(f"{t}/fast_restore.nii.gz", bias_z)
    print(sid, "hirn:", "quelle" if hirn == hirn_q else "hd-bet", "bias: ok", flush=True)
print("fertig", mod)
