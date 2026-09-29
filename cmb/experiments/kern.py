"""The core experiments of the CMB line as ONE repeatable script (user request 2026-09-29; source of truth: REPLIKATION.md).

Each step is the literal command that produced the number in ARBEIT.md / MASTERARBEIT-KERN.md. Steps are idempotent where the
underlying tools are (grids and candidates skip existing cases; trainings are NOT re-run if the run folder has 5 folds).
GPU steps must run through ``grossauftrag --vram 20`` (the language model is stopped for them); ``--queue`` enqueues them
instead of running. Private paths (mb-arena) stay on this machine.

    python -m cmb.experiments.kern --list
    python -m cmb.experiments.kern --steps grid train-basis evaluate-basis csf --dry-run
    python -m cmb.experiments.kern --steps train-basis train-s1 --queue          # GPU via the queue
    python -m cmb.experiments.kern --all --dry-run
"""
from __future__ import annotations
import argparse, glob, os, subprocess, sys

ROOT = "/home/uchralt/data/work/valdo-t2s"; MB = "/home/uchralt/data/work/mb-arena"; EXT = "/home/uchralt/data/extern/momeni-csiro-cmb"
PY = "/home/uchralt/miniconda3/envs/dl/bin/python"; GA = "/home/uchralt/local_agentic_system/system/werkzeuge/grossauftrag"
SYN = "dev/synthseg/{id}_synthseg.nii.gz"; MAN = "dev/manifest_valdo_gitter_d.json"
T = "ergebnisse/turnier"

def trained(run):            # a training run counts as done with 5 fold models
    return len(glob.glob(f"{ROOT}/{run}/fold*/meta.json")) == 5

STEPS = [
  # name, gpu?, done-check, command, what it establishes
  ("grid",            False, lambda: len(glob.glob(f"{ROOT}/dev/gitter_d/*/info.json")) >= 57,
   f"GITTER_C_QUELLE={ROOT}/dev/preproc_roh GITTER_C_ZIEL={ROOT}/dev/gitter_d {PY} code/gitter_c_bauen.py --proc 8",
   "Training grid 0.5x0.5x1 mm from the UNPAINTED preprocessing (no vessel inpainting: +0.136), FRST recomputed"),
  ("manifest",        False, lambda: os.path.exists(f"{ROOT}/{MAN}"),
   f"{PY} -m cmb.transfer.build_valdo_manifest --grid dev/gitter_d --out {MAN}", "57 CV cases, 139 lesions (15 held-out cases excluded)"),
  ("train-basis",     True,  lambda: trained(f"{T}/a03-aniso-e60-gd"),
   f"{PY} code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d", "Baseline a03-aniso, seed 42: 0.585 raw"),
  ("train-s1",        True,  lambda: trained(f"{T}/a03-aniso-s1-e60-gd"),
   f"{PY} code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d --seed 1", "second seed: 0.553 raw (seed noise ~0.08)"),
  ("ensemble",        False, lambda: len(glob.glob(f"{ROOT}/{T}/ens2-a03-aniso-e60-gd/fold*/*_pred_proba.nii.gz")) == 57,
   f"{PY} -m cmb.analysis.ensemble --runs {T}/a03-aniso-e60-gd {T}/a03-aniso-s1-e60-gd --out {T}/ens2-a03-aniso-e60-gd", "two-seed ensemble (deliverable)"),
  ("evaluate-basis",  False, lambda: False,
   f"{PY} -m cmb.transfer.evaluate --run basis=%s/a03-aniso-e60-gd s1=%s/a03-aniso-s1-e60-gd ens2=%s/ens2-a03-aniso-e60-gd --manifest {MAN}" % (T, T, T),
   "raw lesion F1 at the fixed cell 0.3 / 2 mm3"),
  ("csf",             False, lambda: False,
   f'{PY} -m cmb.analysis.operating_point --run basis={T}/a03-aniso-e60-gd s1={T}/a03-aniso-s1-e60-gd ens2={T}/ens2-a03-aniso-e60-gd --manifest {MAN} --synthseg "{SYN}" --workers 6',
   "CSF rule: 0.646 / 0.590 / 0.647 (fixed cell)"),
  ("tournament-check", True, lambda: trained(f"{T}/a01-attunet-e60-gd") and trained(f"{T}/a02-nnunet-e60-gd"),
   f"{PY} code/turnier.py --netz a01-attunet a02-nnunet --epochen 60 --gitter dev/gitter_d", "architecture check on clean data: 0.463 / 0.458 vs 0.585"),
  ("candidates-valdo", False, lambda: os.path.exists(f"{ROOT}/ergebnisse/stage2/a03-aniso-e60-gd-weich/patches.npy"),
   f"{PY} -m cmb.stage2.candidates --predictions {T}/a03-aniso-e60-gd --grid dev/gitter_d --keep-csf --out ergebnisse/stage2/a03-aniso-e60-gd-weich", "278 candidates at 0.15 / 1 mm3"),
  ("train-cat-t2s",   True,  lambda: len(glob.glob(f"{MB}/transfer/a03-aniso-e60-cat-t2s/fold*/meta.json")) == 5,
   f"{PY} code/turnier.py --netz a03-aniso --epochen 60 --manifest {MB}/gitter/manifest_t2star.json --name cat-t2s --basis {MB}/transfer", "own training catalina-T2*: 0.663"),
  ("train-cat-swi",   True,  lambda: len(glob.glob(f"{MB}/transfer/a03-aniso-e60-cat-swi/fold*/meta.json")) == 5,
   f"{PY} code/turnier.py --netz a03-aniso --epochen 60 --manifest {MB}/gitter/manifest_swi.json --name cat-swi --basis {MB}/transfer", "own training catalina-SWI: 0.626"),
  ("candidates-cat",  False, lambda: os.path.exists(f"{MB}/stage2/a03-aniso-e60-cat-swi-weich/patches.npy"),
   f'{PY} -m cmb.stage2.candidates --predictions {MB}/transfer/a03-aniso-e60-cat-t2s --manifest {MB}/gitter/manifest_t2star.json --synthseg "{MB}/synthseg-t2star/robust/{{sid}}_synthseg.nii.gz" --keep-csf --out {MB}/stage2/a03-aniso-e60-cat-t2s-weich && '
   f'{PY} -m cmb.stage2.candidates --predictions {MB}/transfer/a03-aniso-e60-cat-swi --manifest {MB}/gitter/manifest_swi.json --synthseg "{MB}/synthseg-swi/robust/{{sid}}_synthseg.nii.gz" --keep-csf --out {MB}/stage2/a03-aniso-e60-cat-swi-weich',
   "1082 / 870 candidates"),
  ("stage2-pooled",   True,  lambda: False,
   f"{PY} -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich t2star={MB}/stage2/a03-aniso-e60-cat-t2s-weich swi={MB}/stage2/a03-aniso-e60-cat-swi-weich --folds swi={MB}/gitter/manifest_swi_mix.json --models cnn --skip-own --tissue --json {MB}/stage2/pooled_kern.json",
   "shared 3D CNN: SWI +0.045..0.058 vs CSF rule, VALDO / T2* equal"),
  ("stage2-transfer", True,  lambda: False,
   f"{PY} -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich t2star={MB}/stage2/a03-aniso-e60-cat-t2s-weich swi={MB}/stage2/a03-aniso-e60-cat-swi-weich --folds swi={MB}/gitter/manifest_swi_mix.json --models cnn --skip-own --tissue --transfer valdo=t2star,swi --fractions 0 0.25 0.5 1 --json {MB}/stage2/transfer_kern.json",
   "VALDO-only classifier zero-shot: SWI = pooled, T2* -0.022"),
  ("heldout",         True,  lambda: os.path.exists(f"{ROOT}/ergebnisse/validierung/heldout_0928/summary.json"),
   f"echo 'The 15 held-out cases were measured ONCE (2026-09-28, ergebnisse/validierung/heldout_0928). Never again.'", "0.468 pooled / 0.731 without sub-110 / median 0.667"),
  ("momeni-grid",     False, lambda: os.path.exists(f"{EXT}/gitter/manifest_momeni.json"),
   f"{PY} -m cmb.transfer.build_momeni_grid --processes 8", "public external SWI cohort, 57 cases / 146 points"),
  ("momeni-zeroshot", True,  lambda: os.path.exists(f"{EXT}/transfer/zero-shot-valdo-s42/zero_shot.json"),
   f"{PY} -m cmb.transfer.zero_shot --models {T}/a03-aniso-e60-gd --net a03-aniso --manifest {EXT}/gitter/manifest_momeni.json --out {EXT}/transfer/zero-shot-valdo-s42", "VALDO U-Net on Momeni"),
  ("momeni-synthseg", False, lambda: len(glob.glob(f"{EXT}/synthseg/*_synthseg.nii.gz")) >= 57,
   f"{PY} -m cmb.transfer.momeni_synthseg --threads 6", "tissue maps for the CSF rule"),
  ("momeni-score",    False, lambda: False,
   f'{PY} -m cmb.analysis.operating_point --run valdo-s42={EXT}/transfer/zero-shot-valdo-s42 --manifest {EXT}/gitter/manifest_momeni.json --synthseg "{EXT}/synthseg/{{id}}_synthseg.nii.gz" --workers 6',
   "external validation: 0.613 with CSF rule, median per subject 0.667"),
  ("mbnet-prepare",   False, lambda: len(glob.glob(f"{ROOT}/mbnet/valdo/falten/fold4/test/images/*.nii.gz")) > 0,
   f"{PY} -m cmb.baselines.microbleednet_finetune prepare --cohort valdo --processes 6", "original MicrobleedNet inputs (its own preprocessing)"),
  ("mbnet-finetune",  True,  lambda: len(glob.glob(f"{ROOT}/mbnet/valdo/ergebnisse/fold*/lauf.json")) == 5,
   " && ".join(f"{PY} -m cmb.baselines.microbleednet_finetune finetune --cohort valdo --fold {k}" for k in range(5)), "original fine-tuned on our folds (~75 min per fold)"),
  ("mbnet-score",     False, lambda: False,
   f"{PY} -m cmb.baselines.microbleednet_finetune score --cohort valdo", "fair baseline: same cases, same folds, same metric"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="*", default=[]); ap.add_argument("--all", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--queue", action="store_true", help="enqueue GPU steps with grossauftrag instead of running")
    ap.add_argument("--force", action="store_true", help="run even if the done-check says done")
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list or not (a.steps or a.all):
        for n, gpu, done, cmd, what in STEPS:
            print(f"{'[GPU]' if gpu else '[CPU]':5s} {'done ' if done() else 'todo ':5s} {n:18s} {what}")
        return
    todo = names if a.all else a.steps
    for n in todo:
        assert n in names, f"unknown step {n}; see --list"
        gpu, done, cmd, what = next(s[1:] for s in STEPS if s[0] == n)
        if done() and not a.force:
            print(f"== {n}: done (skip; --force to repeat)"); continue
        if gpu:
            full = f"{GA} {'--einreihen ' if a.queue else ''}--vram 20 -- bash -c 'cd {ROOT} && {cmd}'"
        else:
            full = f"cd {ROOT} && {cmd}"
        print(f"== {n}: {what}\n   {full}", flush=True)
        if a.dry_run:
            continue
        r = subprocess.run(full, shell=True)
        if r.returncode != 0:
            raise SystemExit(f"step {n} failed rc={r.returncode}")


if __name__ == "__main__":
    main()
