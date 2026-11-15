#!/usr/bin/env python
"""wk11 step 6/6 -- verify NUMBERS_v3.md against the v3 result JSONs and the manuscript.

For every line of NUMBERS_v3.md:
  (i)  the value at the JSON key path must round to the number as written;
  (ii) the number with its snippet must occur in manuscript.md or supplement.md.
Exit 1 and print the mismatches; exit 0 when all lines pass.
CPU only, read-only: no result file is modified.
"""
import json
import os
import re
import sys

# paper version: the key paths in NUMBERS_v3.md start with the name of one of the two work folders; they are
# resolved through CMB_ROOT (public work folder) and CMB_PRIVATE (private work folder).
_R = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_P = os.environ.get("CMB_PRIVATE", os.path.join(_R, "private"))
# first path component in NUMBERS_v3.md -> folder (set CMB_ROOT_NAME / CMB_PRIVATE_NAME to the names your list uses)
_PREFIX = {os.environ.get("CMB_ROOT_NAME", "root"): _R, os.environ.get("CMB_PRIVATE_NAME", "private"): _P}


def _resolve(relpath):
    head, _, rest = relpath.partition("/")
    return os.path.join(_PREFIX[head], rest) if head in _PREFIX else os.path.join(os.path.dirname(_R), relpath)


PAPER = os.path.join(_R, "paper")
LIST = os.path.join(PAPER, "NUMBERS_v3.md")
TEXTS = [os.path.join(PAPER, "manuscript.md"), os.path.join(PAPER, "supplement.md")]

_JSON_CACHE = {}


def load(relpath):
    if relpath not in _JSON_CACHE:
        with open(_resolve(relpath)) as fh:
            _JSON_CACHE[relpath] = json.load(fh)
    return _JSON_CACHE[relpath]


def walk(obj, keypath):
    """Follow a JSON key path with / as separator; list indices are numbers."""
    cur = obj
    for part in [p for p in keypath.split("/") if p != ""]:
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            if part not in cur:
                raise KeyError(part)
            cur = cur[part]
    return cur


def norm(text):
    """Make a snippet comparable with the manuscript text."""
    text = text.replace("\\*", "*").replace("\\_", "_")
    for dash in ("\u2212", "\u2013", "\u2014"):  # minus, en dash, em dash
        text = text.replace(dash, "-")
    return re.sub(r"\s+", " ", text).strip()


def parse_number(tok):
    try:
        return float(tok.replace("+", ""))
    except ValueError:
        return None


def decimals(tok):
    if "." in tok:
        return len(tok.split(".")[1])
    return 0


def main():
    with open(LIST) as fh:
        raw = fh.read()

    texts = []
    for path in TEXTS:
        if os.path.exists(path):
            with open(path) as fh:
                texts.append(norm(fh.read()))

    n_lines = 0
    fails = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        numtok, snippet, source, keypath = parts[0], parts[1], parts[2], parts[3]
        if parse_number(numtok) is None:  # heading / explanation line, not a number
            continue
        n_lines += 1

        # (i) value at the key path rounds to the number
        try:
            value = walk(load(source), keypath)
            val = float(value)
            tol = 0.5 * 10 ** (-decimals(numtok)) + 1e-9
            ok_val = val == val and abs(val - parse_number(numtok)) <= tol  # noqa: PLR0124
        except Exception as exc:  # missing key, wrong type, unreadable file
            ok_val = False
            value = "ERROR: %s" % exc
        if not ok_val:
            fails.append("%d: value %s != %s at %s (%s)" % (lineno, value, numtok, keypath, source))

        # (ii) number with its snippet occurs in the text
        s = norm(snippet)
        if not any(s in t for t in texts):
            fails.append("%d: snippet not found in text: %r" % (lineno, snippet[:70]))

    print("lines checked: %d" % n_lines)
    print("mismatches: %d" % len(fails))
    for f in fails:
        print("  " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
