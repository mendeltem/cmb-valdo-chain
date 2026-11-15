# Rules for agents and contributors

1. **The paper is frozen.** Tag `paper-v3-2026-11` never moves. The research code that produced the paper's numbers
   (the `cmb/` research packages, `cmb/metrics.py`, `cmb/paths.py`, `code/`, `scripts/`, `pipeline/`, `dev/`) is not
   changed, not even reformatted; `tests/test_frozen.py` checks it byte for byte. Work happens in the command-line and
   library layer (`cmb/cli.py`, `commands.py`, `predict_cli.py`, `api.py`, `checks.py`, `nifti.py`, ...).
2. **No quick fixes.** Find and fix the cause, then add the test or check that would have caught it.
3. **No new Markdown files** without asking the maintainer first.
4. **Comments say WHY** (a method choice, the measurement behind it, the paper step it reproduces). Comments that
   restate the code are removed.
5. **Data-agnostic.** No cohort names, subject ids other than public VALDO ones, or machine paths (no absolute or
   home-relative paths of one machine); locations come from `cmb/paths.py` / environment variables.
   `tests/test_privacy.py` checks this.
6. **Run the checks before every push** (CI runs the same on GitHub):

   ```bash
   ruff check . && ruff format --check . && pyright && lint-imports
   python -m pytest -q tests -m "not valdo"
   ```

   With VALDO linked locally, also `python -m pytest -q tests -m valdo`.
