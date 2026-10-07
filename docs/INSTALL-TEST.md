# Install test (fresh clone, fresh conda environment)

Run on 2026-10-07 on Ubuntu 24.04, conda 25.11, from a fresh clone, exactly the README commands (CPU build of torch).
First attempt with the old `requirements-lock.txt` failed at once: `Invalid requirement: 'torch_cluster==1.6.3%2Bpt26cu124'` (pip cannot read the URL-encoded build tags, and the +cu124 wheels are not on PyPI). Fix: `env/requirements.txt` (runtime only, torch installed separately from the PyTorch index); the old file is kept as `requirements-full-record.txt`. Second attempt: everything installed, the self-test failed only on a missing `tqdm` (used by the FRST self-test) -- added. Third run of the self-test:

```
SELFTEST PASSED
ok    FRST (1.6 s)
ok    lesion metric (0.0 s)
ok    U-Net forward (0.7 s)
ok    imports (0.3 s)
ok    dry run of all steps (0.0 s)
SELFTEST PASSED
```

Installed versions:
```
monai==1.5.2
nibabel==5.4.2
numpy==2.3.5
pillow==12.3.0
scikit-image==0.26.0
scikit-learn==1.9.0
scipy==1.18.1
simpleitk==2.5.6
torch==2.6.0+cpu
tqdm==4.70.0
```
