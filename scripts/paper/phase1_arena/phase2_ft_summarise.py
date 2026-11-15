"""wk2b CMB Phase II -- P->FT (pooled + fine-tune) summary + figure.

Reads the per-case accepted-count JSONs for the P->FT arm (phase2_ft_seed{0,1,2}.json)
and the S/S and S/P arms (phase2_seed{0,1,2}.json), produces:
  - MB/paper/phase2_finetune_summary.json  (delta F1 P->FT vs S/P and P->FT vs S/S, per cohort + overall, case-level bootstrap 95% CI)
  - <repo>/paper/fig2d_pooled_finetune.png  (F1 per cohort for S/S, S/P, P->FT with 95% CI; English)

Run from the repository root:  python scripts/paper/phase1_arena/phase2_ft_summarise.py
"""
from __future__ import annotations
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json, glob, os
import numpy as np

sys_path = _R
import sys
sys.path.insert(0, sys_path)
from cmb.stage2.tset_summary import f1_arr, SIGMA_SEED_FLOOR  # noqa

ROOT = sys_path
O = f"{ROOT}/ergebnisse/phase2"
MB_PAPER = _P + "/paper"
P = f"{ROOT}/paper"
COHORTS = ["swi", "t2star", "valdo"]
DRAWS, BSEED = 4000, 0

SS = "own tset-plain"
SP = "pooled tset-plain"
FT = "pooled-ft tset-plain"


def load_seeds(pattern: str) -> dict:
    """seed_offset -> mode -> cohort -> case -> (tp, fp, fn)."""
    seeds = {}
    for path in sorted(glob.glob(pattern)):
        data = json.load(open(path))
        s = int(data.get("seed_offset", 0))
        if "status" in data and "pending" in data.get("status", ""):
            continue   # skip placeholder
        if s not in seeds:
            seeds[s] = {}
        for mode, per in data["per_case"].items():
            seeds[s][mode] = {c: {k: tuple(v) for k, v in cases.items()} for c, cases in per.items()}
    return seeds


def mean_f1(seeds_by_seed, mode, cohort, case_keys, seeds):
    """Mean per-case (tp,fp,fn) over seeds, then f1_arr."""
    arr = np.array([seeds_by_seed[s][mode][cohort][k] for s in seeds for k in case_keys], float).reshape(-1, 3)
    mean = arr.reshape(len(seeds), -1, 3).mean(axis=0)
    return f1_arr(mean)


def delta_f1_ci(seeds_ft, seeds_ref, mode_ref, cohort, case_keys, seeds, draws=DRAWS, seed=BSEED):
    """Case-level stratified bootstrap for delta F1 (P->FT - ref) on one cohort."""
    rng = np.random.RandomState(seed)
    n = len(case_keys)
    # mean per-case counts over seeds
    ft_arr = np.array([seeds_ft[s][FT][cohort][k] for s in seeds for k in case_keys], float).reshape(len(seeds), n, 3)
    ref_arr = np.array([seeds_ref[s][mode_ref][cohort][k] for s in seeds for k in case_keys], float).reshape(len(seeds), n, 3)
    ft_mean = ft_arr.mean(axis=0)
    ref_mean = ref_arr.mean(axis=0)
    delta = f1_arr(ft_mean) - f1_arr(ref_mean)
    boots = np.zeros(draws)
    for d in range(draws):
        ix = rng.choice(n, size=n)
        boots[d] = f1_arr(ft_mean[ix]) - f1_arr(ref_mean[ix])
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return round(float(delta), 4), [round(ci[0], 4), round(ci[1], 4)]


def main():
    ft_seeds = load_seeds(f"{O}/phase2_ft_seed*.json")
    sp_seeds = load_seeds(f"{O}/phase2_seed*.json")
    all_seeds = sorted(set(ft_seeds) & set(sp_seeds))
    if not all_seeds:
        raise SystemExit("no common seeds between phase2_ft_seed*.json and phase2_seed*.json")

    # build case keys per cohort from SP (the control)
    ref0 = sp_seeds[all_seeds[0]][SP]
    case_keys = {c: sorted(ref0[c]) for c in COHORTS}

    # F1 per arm per cohort
    f1s = {}
    for label, seeds_map, mode in [("S/S", sp_seeds, SS), ("S/P", sp_seeds, SP), ("P->FT", ft_seeds, FT)]:
        f1s[label] = {}
        for c in COHORTS:
            f1s[label][c] = round(mean_f1(seeds_map, mode, c, case_keys[c], all_seeds), 4)
        # overall (all cases pooled)
        all_k = [k for c in COHORTS for k in case_keys[c]]
        # build a combined per-case array
        arr = np.array([seeds_map[s][mode][c][k] for s in all_seeds for c in COHORTS for k in case_keys[c]], float)
        f1s[label]["all"] = round(f1_arr(arr.reshape(len(all_seeds), -1, 3).mean(axis=0)), 4)

    # delta F1 with case-level bootstrap 95% CI
    comparisons = {}
    for ref_label, ref_mode in [("vs_S/P", SP), ("vs_S/S", SS)]:
        comparisons[ref_label] = {}
        for c in COHORTS + ["all"]:
            if c == "all":
                # pooled bootstrap: combine all cohorts
                ft_arr = np.array([ft_seeds[s][FT][cc][k] for s in all_seeds for cc in COHORTS for k in case_keys[cc]], float)
                ref_arr = np.array([sp_seeds[s][ref_mode][cc][k] for s in all_seeds for cc in COHORTS for k in case_keys[cc]], float)
                ft_m = ft_arr.reshape(len(all_seeds), -1, 3).mean(axis=0)
                ref_m = ref_arr.reshape(len(all_seeds), -1, 3).mean(axis=0)
                n = len(ft_m)
                rng = np.random.RandomState(BSEED)
                boots = np.zeros(DRAWS)
                for d in range(DRAWS):
                    ix = rng.choice(n, size=n)
                    boots[d] = f1_arr(ft_m[ix]) - f1_arr(ref_m[ix])
                delta = float(f1_arr(ft_m) - f1_arr(ref_m))
                ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                comparisons[ref_label][c] = dict(delta=round(delta, 4), ci95=[round(ci[0], 4), round(ci[1], 4)])
            else:
                delta, ci = delta_f1_ci(ft_seeds, sp_seeds, ref_mode, c, case_keys[c], all_seeds)
                comparisons[ref_label][c] = dict(delta=delta, ci95=ci)

    summary = dict(
        title="wk2b CMB Phase II -- P->FT (pooled + fine-tune) vs S/P and S/S, stage 2 tset-plain, 5-fold CV, rules v2",
        arm=FT,
        control=SP,
        n_seeds=len(all_seeds),
        seeds=all_seeds,
        cohorts=COHORTS,
        f1={lab: {c: f1s[lab][c] for c in COHORTS + ["all"]} for lab in ["S/S", "S/P", "P->FT"]},
        comparisons=comparisons,
        note="Exploratory arm added after wk2 pre-registration. P->FT: pre-train on pooled candidates (all cohorts, fold!=k) "
             "+ fine-tune on target D's own candidates (fold!=k, half epochs, lr/2). "
             "Reported as an additional comparison, never as the primary test.",
    )

    out = f"{MB_PAPER}/phase2_finetune_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out}")

    # figure: fig2d
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    except ImportError:
        print("matplotlib not available, skipping figure")
        return

    def f1_ci(seeds_map, mode, cohort, keys, seeds):
        """Bootstrap 95 % CI of the per-case F1 of one arm on one cohort."""
        arr = np.array([seeds_map[s][mode][cohort][k] for s in seeds for k in keys], float).reshape(len(seeds), len(keys), 3).mean(axis=0)
        rng = np.random.RandomState(BSEED)
        boots = np.zeros(DRAWS)
        for d in range(DRAWS):
            ix = rng.choice(len(keys), size=len(keys))
            boots[d] = f1_arr(arr[ix])
        return float(f1_arr(arr)), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    x = np.arange(len(COHORTS))
    n_arms = 3
    w = 0.8 / n_arms
    colors = ["#2b6ca3", "#d9822b", "#5a8f3c"]
    labels = ["S/S (separate stage 2)", "S/P (pooled stage 2)", "P->FT (pooled + fine-tune)"]
    arm_keys = ["S/S", "S/P", "P->FT"]
    seedmaps = {"S/S": sp_seeds, "S/P": sp_seeds, "P->FT": ft_seeds}
    modes = {"S/S": SS, "S/P": SP, "P->FT": FT}
    for i, ak in enumerate(arm_keys):
        for j, c in enumerate(COHORTS):
            v, (lo, hi) = f1_ci(seedmaps[ak], modes[ak], c, case_keys[c], all_seeds)
            ax.errorbar(x[j] + (i - 1) * w, v, yerr=[[v - lo], [hi - v]], fmt="none",
                        ecolor=colors[i], elinewidth=1.5, capsize=5, capthick=1.5)
            ax.plot(x[j] + (i - 1) * w, v, "o", color=colors[i], ms=7,
                    label=labels[i] if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(COHORTS)
    ax.set_ylabel("Lesion-level F1")
    ax.set_title("P->FT (pooled + fine-tune) vs S/P and S/S\nStage 2 tset-plain, 5-fold CV, 3 seeds (95 % CI)\nExploratory arm, wk2b -- not the primary test")
    ax.legend(frameon=False, fontsize=10, loc="center right")
    ax.set_ylim(0.4, 0.8)
    fig.tight_layout()
    out_png = f"{P}/fig2d_pooled_finetune.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
