# Install test (fresh clone, fresh conda environment)

## 2026-11-15, tag paper-v3-2026-11

Fresh clone of commit c3c827c into a temporary folder, a NEW conda environment (`./install.sh --name cmbvc-fresh2-1115
--torch cpu`), then every ```` ```bash ```` block of the README verbatim (skill install-from-zero-check), on the CPU, with
`CMB_VALDO` pointing at a VALDO copy and the weights served from a local copy of the release assets (`CMB_MODEL_URL`;
the download from the GitHub release itself was checked after the release was published, see below).

| block | command | result |
|---|---|---|
| install | `./install.sh` (conda env, PyTorch 2.6.0 CPU, env/requirements.txt, pip install -e, weights, `cmb selftest`) | OK, 41 s (pip cache warm) |
| 2 | `cmb selftest` | OK, 2 s |
| 3 | `bash scripts/link_paths.sh` | OK |
| 4 | `cmb predict -i data/valdo2021/Task2/sub-101/...` | OK, 641 s on a loaded CPU |
| 5 | `python -m cmb.synthetic ...; cmb preprocess ...` | OK, 1 s |
| 6 | `cmb train ...; cmb evaluate ...` | OK, 77 s |
| 7 | `cmb fine_tune` (all, norm_out, calibrate_only), `cmb predict -m` | OK, 109 s |
| 8 | `cmb cross_validate ... -fold 5` | OK, 201 s |
| 9-11 | `python examples/python/tutorial_{predict,train,finetune}.py` | OK, 23 s / 527 s / 74 s |
| 12 | `python -m pytest -q tests/test_cli.py tests/test_privacy.py` | OK |
| 13 | `bash configs/smoke.sh cpu` | OK, 234 s |

The first attempt (commit d9dcedd, environment cmbvc-fresh-1115) failed at `cmb selftest`: the FRST self-check imported
the original MicrobleedNet package for a printed comparison. Fixed in c3c827c (the comparison is skipped without the
package), with `tests/test_selftest.py`, which fails on the old file and passes on the new one.
The full suite (`python -m pytest -q tests`) passed in a development environment built by the same installer.

## 2026-10-07 (earlier version)

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
