"""The three Python tutorials run as they are (CPU), from the repository root and from another directory, with the
paths of the committed .env, and stay within 10 lines of code."""
import ast
import os
import subprocess
import sys

import pytest

from conftest import ENV, REPO

TUTORIALS = ["tutorial_predict.py", "tutorial_train.py", "tutorial_finetune.py"]


def code_lines(path):
    """Lines of code: blank lines, comments and the module docstring are not counted."""
    src = open(path).read()
    tree = ast.parse(src)
    doc = set()
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant):
        doc = set(range(tree.body[0].lineno, tree.body[0].end_lineno + 1))
    return sum(1 for n, line in enumerate(src.splitlines(), 1)
               if n not in doc and line.strip() and not line.strip().startswith("#"))


@pytest.mark.parametrize("name", TUTORIALS)
def test_at_most_ten_lines_and_no_paths(name):
    path = os.path.join(REPO, "examples", "python", name)
    assert code_lines(path) <= 10
    body = open(path).read().split('"""', 2)[-1]
    assert "/" not in body.replace("->", ""), "paths belong in .env"


def test_env_file_is_committed_relative_and_public():
    from cmb.envfile import env
    text = open(os.path.join(REPO, ".env")).read()
    assert "/home" not in text and "~" not in text
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            assert not line.split("=", 1)[1].startswith("/"), line
    r = subprocess.run(["git", "check-ignore", ".env"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 1, ".env must not be ignored by git"
    assert env().CMB_SCAN.startswith(REPO)


@pytest.mark.parametrize("name", TUTORIALS)
@pytest.mark.parametrize("where", ["repo", "elsewhere"])
def test_tutorial_runs(name, where, tmp_path):
    cwd = REPO if where == "repo" else str(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(REPO, "examples", "python", name)], cwd=cwd, env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "->" in r.stdout.strip().splitlines()[-1]
