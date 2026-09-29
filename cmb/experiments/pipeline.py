"""Shared entry point of the numbered scripts in ``pipeline/``: each script names its steps of ``kern.STEPS``
and forwards ``--dry-run``, ``--queue``, ``--force`` and ``--list`` to the runner, so the done-checks stay in one place."""
from __future__ import annotations
import sys
from cmb.experiments import kern


def run(steps: list[str], doc: str | None = None) -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print((doc or "").strip()); print("\nsteps:", " ".join(steps))
        print("options: --dry-run (print commands only)  --queue (GPU steps via the grossauftrag queue)  --force  --list"); return
    if "--list" in argv:
        sys.argv = [sys.argv[0], "--list"]; kern.main(); return
    sys.argv = [sys.argv[0], "--steps", *steps, *[a for a in argv if a in ("--dry-run", "--queue", "--force")]]
    kern.main()
