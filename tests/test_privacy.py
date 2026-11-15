"""The public repository stays data-agnostic: no machine paths, no in-house cohort names or subject ids."""
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = re.compile(r"/home/[a-z]|uchralt|sc-projects|sc-scratch|s-sc-frontend|csw-|c2s-|\bc2s\b|\bcsw\b|\bsub-t\d|\bPDD\d"
                     r"|~/(?!\.cache|\.local)|expanduser\(\"~/|catalina|charit[eé]|\balita\b|belove|demdas|mb-arena|valdo-t2s", re.I)


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO, capture_output=True, text=True).stdout
    return [f for f in out.split() if os.path.isfile(os.path.join(REPO, f))]


def test_no_private_names_or_paths():
    hits = []
    for f in tracked_files():
        if f == "tests/test_privacy.py" or f.endswith((".png", ".svg", ".pt", ".npy", ".gz")):
            continue
        try:
            text = open(os.path.join(REPO, f), encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{f}:{n}: {line.strip()[:120]}")
    assert not hits, "\n".join(hits[:40])


def test_only_public_valdo_ids():
    bad = []
    for f in tracked_files():
        if f.endswith((".py", ".sh", ".md", ".json", ".txt", ".yml")):
            for m in re.finditer(r"\bsub-([A-Za-z]*\d+)", open(os.path.join(REPO, f), encoding="utf-8", errors="ignore").read()):
                v = m.group(1)
                # VALDO ids sub-1xx/2xx/3xx, cohort prefixes sub-1/2/3, synthetic sub-00x; nothing else
                if not (v.isdigit() and (len(v) < 3 or (len(v) == 3 and v[0] in "0123"))):
                    bad.append(f"{f}: sub-{v}")
    assert not bad, bad[:20]
