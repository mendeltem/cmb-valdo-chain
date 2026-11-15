"""The research code that produced the paper's numbers stays as it ran (AGENTS.md), and nothing imports from the
data folders.

* every file of the research code (the cmb/ research packages, cmb/metrics.py, cmb/paths.py, code/, scripts/,
  pipeline/, dev/) has the SHA-256 recorded in tests/frozen.sha256; a new file there fails too. The recorded state
  is the tag paper-v3-2026-11 plus the documented portability fixes after it (CHANGES.md).
* no Python file puts dev/ or the private work folder on sys.path (import-linter sees only the cmb package; code/
  and scripts/ are imported by path).

An agreed change of research code: regenerate with ``python tests/test_frozen.py --update`` and give the reason in
CHANGES.md.
"""

import ast
import hashlib
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(REPO, "tests", "frozen.sha256")
FROZEN = (
    "code/",
    "scripts/",
    "pipeline/",
    "dev/",
    "cmb/analysis/",
    "cmb/baselines/",
    "cmb/experiments/",
    "cmb/maskfix/",
    "cmb/models/",
    "cmb/review/",
    "cmb/stage2/",
    "cmb/synthesis/",
    "cmb/teaching/",
    "cmb/training/",
    "cmb/transfer/",
    "cmb/tests/",
    "cmb/metrics.py",
    "cmb/paths.py",
)
DATA_ON_PATH = re.compile(r"""["'/]dev\b|\bdev["'/]|private|PRIVATE|\bMB\b""")


def tracked():
    r = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    return [f for f in r.stdout.splitlines() if os.path.isfile(os.path.join(REPO, f))]


def digest(path):
    with open(os.path.join(REPO, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def frozen_files():
    return sorted(f for f in tracked() if f.startswith(FROZEN))


def recorded():
    with open(RECORD) as f:
        pairs = [line.split(None, 1) for line in f.read().splitlines() if line.strip()]
    return {path.strip(): sha for sha, path in pairs}


def test_research_code_is_unchanged():
    want = recorded()
    have = {f: digest(f) for f in frozen_files()}
    changed = sorted(f for f in want if f in have and have[f] != want[f])
    missing = sorted(set(want) - set(have))
    new = sorted(set(have) - set(want))
    assert not (changed or missing or new), (
        f"research code changed: {changed[:10]}, removed: {missing[:10]}, new: {new[:10]} -- it is frozen at "
        "paper-v3-2026-11 (AGENTS.md); an agreed change: python tests/test_frozen.py --update + reason in CHANGES.md"
    )


def test_nothing_imports_from_the_data_folders():
    hits = []
    for f in tracked():
        if not f.endswith(".py"):
            continue
        with open(os.path.join(REPO, f), encoding="utf-8") as fh:
            src = fh.read()
        for node in ast.walk(ast.parse(src)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("insert", "append")
                and ast.unparse(node.func.value) == "sys.path"
            ):
                arg = ast.unparse(node.args[-1]) if node.args else ""
                if DATA_ON_PATH.search(arg):
                    hits.append(f"{f}:{node.lineno}: sys.path.{node.func.attr}({arg})")
    assert not hits, hits


if __name__ == "__main__":
    if sys.argv[1:] == ["--update"]:
        with open(RECORD, "w") as out:
            out.writelines(f"{digest(f)}  {f}\n" for f in frozen_files())
        print(f"{RECORD}: {len(frozen_files())} files")
