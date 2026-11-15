"""wk1b: MedNet-CMB v2-Auswertung -- exakt das Post-Processing der eigenen Kette.

Wiederverwendet die Wahrscheinlichkeitskarten des GPU-Laufs (mednet-{swi,t2star}/prob/*.npz,
Key 'probabilities', Shape (2, z, y, x), float32; Kanal 0 = gesaettigtes Background-Label,
Kanal 1 = CMB-Wahrscheinlichkeit, auf dem Original-Bildgitter). Kein neuer GPU-Lauf.
Logik identisch zu v2_postprocess.py (SHIVA): 26er-Komponenten, Groessenfilter
[MIN_COMPONENT_VOXELS, MAX_COMPONENT_VOXELS], Schwellengitter 0.05..0.95 (nie 0),
author (0.5) + nested (gepoolte Touch-F1 ueber die 4 Trainingsfalten),
case-bootstrap 95 % CI, Subjektlevel (Spearman, ICC(3,1), 4-CMB-Schwelle).

Nur Summen, keine Fallkennungen in der Ausgabe.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import nibabel as nib
import v2_postprocess as V   # prepare_case, case_counts, choose_nested, score, GRID, MIN/MAX, ROOT, FAELLE


def load_prob(cohort, case_id, root):
    """npz 'probabilities' (2, z, y, x) -> Kanal 1 (CMB) auf (x, y, z) = Bildgitter."""
    p = np.load(os.path.join(root, f"mednet-{cohort}", "prob", f"{case_id}.npz"))["probabilities"]
    return p[1].transpose(2, 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["swi", "t2star"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Faelle (Probelauf)")
    args = ap.parse_args()
    t0 = time.time()
    faelles = json.load(open(V.FAELLE))[args.cohort]["faelle"]
    if args.limit:
        faelles = faelles[:args.limit]
    cases = []
    for c in faelles:
        p = load_prob(args.cohort, c["id"], V.ROOT)
        r = np.asarray(nib.load(c["maske"]).dataobj) > 0
        case = V.prepare_case(p, r)
        case["fold"] = int(c["fold"])
        cases.append(case)
        print(f"{c['id']}: n_ref={case['n_ref']}, n_comp(t=0.05)={len(case['per_t'][V.GRID[0]])}", flush=True)
    t1 = time.time()
    counts = {(i, t): V.case_counts(case, t) for i, case in enumerate(cases) for t in V.GRID}
    print(f"precount {len(cases)}x{len(V.GRID)}: {round(time.time() - t1)} s", flush=True)
    t_nested, f1_train = V.choose_nested(cases, counts)
    for case, t in zip(cases, t_nested):
        case["t_nested"] = t
    res = dict(cohort=args.cohort, model="mednet-cmb", n_cases=len(cases),
               postprocessing="26-connected; keep MIN<=voxels<=MAX (4..2100); "
                              "MIN aus cmb/stage2/candidates.py voxels*0.25<min_mm3(1.0); MAX_COMPONENT_VOXELS=2100",
               threshold_grid=V.GRID,
               author_threshold=V.score(cases, lambda c: 0.5, "as-downloaded (author threshold 0.5, postprocessed)"),
               nested=V.score(cases, lambda c: c["t_nested"], "nested (per-case threshold from 4 training folds, postprocessed)"),
               thresholds_nested={str(t): f for t, f in zip(t_nested, f1_train)})

    def pool_comp(key):
        agg = {k: 0 for k in ("n_components", "n_dropped_small", "n_dropped_large", "n_kept", "kept_voxels")}
        sizes = []
        for c in cases:
            t = 0.5 if key == "a" else c["t_nested"]
            s = V.comp_stats_at(c, t)
            for k in agg:
                agg[k] += s[k]
            for cc in c["per_t"][t]:
                if V.MIN_COMPONENT_VOXELS <= cc["size"] <= V.MAX_COMPONENT_VOXELS:
                    sizes.append(cc["size"])
        agg["kept_median_voxels"] = round(float(np.median(sizes)), 1) if sizes else 0.0
        agg["kept_p95_voxels"] = round(float(np.percentile(sizes, 95)), 1) if sizes else 0.0
        return agg
    res["components_author"] = pool_comp("a")
    res["components_nested"] = pool_comp("n")
    res["dauer_s"] = round(time.time() - t0)
    out = os.path.join(V.ROOT, f"v2_mednet_{args.cohort}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("author :", json.dumps({k: res["author_threshold"][k] for k in ("f1", "precision", "sensitivity", "fp_per_case", "f1_ci95")}))
    print("nested :", json.dumps({k: res["nested"][k] for k in ("f1", "precision", "sensitivity", "fp_per_case", "f1_ci95")}))
    print("written:", out)


if __name__ == "__main__":
    main()
