"""Keep only the candidates of a synthetic run that sit at INSERTED objects (2026-10-25, Arena 2).

The candidates of a modified host copy also contain the host's own lesions and false alarms a second time (with recomputed FRST
and probability). As training-only extras (cmb.stage2.pooled --extra) those near-duplicates would double the weight of the real
training candidates; only the inserted microbleeds (positive) and veins (negative) carry new information. Matching as in
cmb.synthesis.realism_probe (centre within 3 mm of an inserted bleed, 4 mm of an inserted vein). Inserted bleeds keep their
'touched' list; vein candidates must have an empty one (a vein candidate touching a reference lesion is dropped and counted).

    python -m cmb.synthesis.keep_inserted --candidates <out>/candidates --log <out>/insert_log.json --out <out>/candidates-inserted
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cmb.synthesis.realism_probe import inserted_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True); parser.add_argument("--log", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    info = json.load(open(f"{args.candidates}/candidates.json"))
    patches = np.load(f"{args.candidates}/patches.npy", mmap_mode="r")
    log = json.load(open(args.log))["objects"]
    bleeds = set(inserted_candidates(info["candidates"], log, "bleed")); veins = set(inserted_candidates(info["candidates"], log, "vein"))
    keep, dropped = [], 0
    for i, r in enumerate(info["candidates"]):
        if i in bleeds and r["touched"]:
            keep.append(i)
        elif i in veins and not r["touched"] and i not in bleeds:
            keep.append(i)
        elif i in veins or i in bleeds:
            dropped += 1
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/patches.npy", np.asarray(patches[keep]) if keep else np.zeros((0,) + patches.shape[1:], patches.dtype))
    out = dict(info, candidates=[info["candidates"][i] for i in keep], kept_from=args.candidates, dropped_ambiguous=dropped)
    json.dump(out, open(f"{args.out}/candidates.json", "w"))
    pos = sum(bool(info["candidates"][i]["touched"]) for i in keep)
    print(f"kept {len(keep)} of {len(info['candidates'])} candidates ({pos} at inserted bleeds, {len(keep) - pos} at inserted veins, "
          f"{dropped} ambiguous dropped) -> {args.out}")


if __name__ == "__main__":
    main()
