"""Shared fixtures: a tiny synthetic cohort (cmb.synthetic), its grid (cmb preprocess) and a 1-epoch model (cmb train).
Everything runs on the CPU in a temporary folder; no real data. VALDO tests are skipped without VALDO (CMB_VALDO)."""

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(
    os.environ,
    CV5_DEV="cpu",
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"),
    CMB_ROOT=REPO,
)


def cmb(*args, check=True, cwd=None):
    """Run the installed command line exactly as a user would (python -m cmb == the `cmb` entry point)."""
    r = subprocess.run(
        [sys.executable, "-m", "cmb", *map(str, args)], env=ENV, cwd=cwd or REPO, capture_output=True, text=True
    )
    if check and r.returncode != 0:
        raise AssertionError(
            f"cmb {' '.join(map(str, args))} failed rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
        )
    return r


@pytest.fixture(scope="session")
def cohort(tmp_path_factory):
    from cmb import synthetic

    d = tmp_path_factory.mktemp("cohort")
    raw, ids = synthetic.write_cohort(str(d), n=6)
    return dict(dir=str(d), raw=raw, ids=ids)


@pytest.fixture(scope="session")
def grid(cohort, tmp_path_factory):
    out = tmp_path_factory.mktemp("grid")
    cmb(
        "preprocess",
        "-i",
        cohort["raw"],
        "-ir",
        "_T2starw",
        "-f",
        "True",
        "-b",
        cohort["raw"],
        "-br",
        "_brainmask",
        "-l",
        cohort["raw"],
        "-lr",
        "_cmb",
        "-o",
        out,
        "-fold",
        "2",
    )
    return str(out)


@pytest.fixture(scope="session")
def model(grid, tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    cmb("train", "-i", grid, "-l", grid, "-m", out, "-ep", "1", "--patches", "8", "--seeds", "42", "--device", "cpu")
    return str(out)
