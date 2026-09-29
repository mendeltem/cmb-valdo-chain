"""Follow the positive:negative sampling ratio as long as it keeps helping.

User request 2026-09-20: "if 1:2 is better, can you also add 1:3, 1:4, ...?"  This script is queued
AFTER the 1:2 and 1:1 runs. It looks at the finished runs and queues the next ratio by a rule fixed
in advance, then queues itself again behind it:

    share of patches that contain a lesion:  0.60 (recipe, 1.5:1)  0.50 (1:1)  0.333 (1:2)
                                             0.25 (1:3)  0.20 (1:4)  0.167 (1:5)  0.143 (1:6)

Rule: walk towards fewer lesion patches while the pooled F1 of the newest ratio is higher than
that of the best ratio before it (paired over the same 57 cases; the bootstrap interval is printed
but NOT required to exclude zero -- one step costs 1.7 h, and the curve as a whole is the result).
Stop at the first step that does not improve, or at 1:6. If neither 1:1 nor 1:2 beats the recipe,
try the other direction once (2:1 = 0.667). Everything is appended to protokoll.md.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
import time

import numpy as np

ROOT = "/home/uchralt/data/work/valdo-t2s"
RUNS = f"{ROOT}/ergebnisse/turnier"
QUEUE_TOOL = "/home/uchralt/local_agentic_system/system/werkzeuge/grossauftrag"
PYTHON = "/home/uchralt/miniconda3/envs/dl/bin/python"
LADDER = [0.6, 0.5, 0.333, 0.25, 0.2, 0.167, 0.143]          # towards more negatives
NAMES = {0.6: "1.5:1 (recipe)", 0.5: "1:1", 0.333: "1:2", 0.25: "1:3", 0.2: "1:4", 0.167: "1:5", 0.143: "1:6", 0.667: "2:1"}


def folder(share: float) -> str:
    return f"{RUNS}/a03-aniso-e60-gd" if share == 0.6 else f"{RUNS}/a03-aniso-e60-l{round(share * 100)}-gd"


def rows(share: float):
    metas = glob.glob(f"{folder(share)}/fold*/meta.json")
    if len(metas) < 5 or not all(json.load(open(m)).get("ok") for m in metas):
        return None
    return {json.load(open(f))["id"]: json.load(open(f)) for f in glob.glob(f"{folder(share)}/fold*/*_pred_meta.json")}


def f1(r) -> float:
    tp = sum(x["tp"] for x in r); fp = sum(x["fp"] for x in r); fn = sum(x["fn"] for x in r)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def paired(a, b):
    ids = sorted(set(a) & set(b)); A = [a[i] for i in ids]; B = [b[i] for i in ids]
    rng = np.random.RandomState(0); d = []
    for _ in range(5000):
        ix = rng.randint(len(ids), size=len(ids)); d.append(f1([A[j] for j in ix]) - f1([B[j] for j in ix]))
    return f1(A) - f1(B), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def queue(*command: str) -> None:
    subprocess.run([QUEUE_TOOL, "--einreihen", "--vram", "20", "--gib", "14", "--", *command], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


def main() -> None:
    done = {s: rows(s) for s in LADDER + [0.667]}
    done = {s: r for s, r in done.items() if r}
    lines = [f"**Ratio chain {time.strftime('%d.%m. %H:%M')} (cmb/analysis/ratio_chain.py):**"]
    for s in sorted(done, reverse=True):
        extra = ""
        if s != 0.6 and 0.6 in done:
            delta, lo, hi = paired(done[s], done[0.6]); extra = f", gegen Rezept {delta:+.3f} [{lo:+.3f}; {hi:+.3f}]"
        lines.append(f"  {NAMES[s]:16s} F1 {f1(list(done[s].values())):.3f}{extra}")

    next_share = None
    walked = [s for s in LADDER if s in done]                     # contiguous part of the ladder that exists
    if len(walked) >= 3 and walked == LADDER[:len(walked)]:
        newest, before = walked[-1], walked[:-1]
        best_before = max(before, key=lambda s: f1(list(done[s].values())))
        if f1(list(done[newest].values())) > f1(list(done[best_before].values())) and len(walked) < len(LADDER):
            next_share = LADDER[len(walked)]
        elif len(walked) == 3 and best_before == 0.6 and f1(list(done[0.5].values())) < f1(list(done[0.6].values())) and 0.667 not in done:
            next_share = 0.667                                   # neither 1:1 nor 1:2 helped: look the other way once
    if next_share:
        lines.append(f"  -> next step queued: {NAMES[next_share]} (share {next_share})")
        queue(PYTHON, f"{ROOT}/code/turnier.py", "--netz", "a03-aniso", "--epochen", "60", "--gitter", "dev/gitter_d",
              "--anteil-laesion", str(next_share))
        queue(PYTHON, "-m", "cmb.analysis.ratio_chain")
    else:
        best = max(done, key=lambda s: f1(list(done[s].values())))
        lines.append(f"  -> chain finished. Best ratio: {NAMES[best]} (F1 {f1(list(done[best].values())):.3f}).")
    text = "\n".join(lines)
    print(text)
    open(f"{ROOT}/protokoll.md", "a", encoding="utf-8").write("\n" + text + "\n")


if __name__ == "__main__":
    sys.exit(main())
