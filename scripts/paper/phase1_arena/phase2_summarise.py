"""wk2 CMB Phase II -- POOLING FACTORIAL: summary + figures.

Reads the per-case accepted-count JSONs written by cmb.stage2.pooled (one file per seed
offset) and produces the three Phase-2 summary files + their figures, using the SAME
rules-v2 statistics as cmb.stage2.tset_summary (stratified bootstrap over cases,
pooled F1 on 'all', max-T simultaneous lower bound) plus the PRE-REGISTERED
superiority / non-inferiority contrasts written verbatim into ARBEIT.md wk2.

Arms available on the CPU stage-2 runs (tset-plain):
  SS  = own tset-plain        (separate stage 2 per dataset)
  SP  = pooled tset-plain     (mixed stage 2 = CONTROL)
  M_  = matched tset-plain <D>  (size-matched mixed control, evaluated on D only)
  Cprobo / Cprobp  = own / pooled tset-plain -prob      (mechanism, probability only)
  Cpfro  / Cpfro  ... = own / pooled tset-plain -prob-frst (mechanism, FRST+probability)
  Call    = the SS/SP runs themselves (image + FRST + probability)
Stage-1 unified arms (PS, PP, PTFT) are GPU-only -> reported as pending until the queue
runs deliver their candidate sets.

Run from the repository root:  python scripts/paper/phase1_arena/phase2_summarise.py
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
import json, os, sys, glob
import numpy as np

sys.path.insert(0, _R)
from cmb.stage2.tset_summary import f1_arr, DELTA_MIN, FAMILY_LEVEL  # noqa

ROOT = _R
O = f"{ROOT}/ergebnisse/phase2"
MB_PAPER = _P + "/paper"
P = f"{ROOT}/paper"
COHORTS = ["swi", "t2star", "valdo"]
DRAWS, BSEED = 4000, 0
NEG_TRANSFER = 0.0     # dataset whose 95 % CI lies entirely below 0
NI_MARGIN = -0.02      # pre-registered non-inferiority margin

# (json mode label in pooled output, short tag)
SS = "own tset-plain"
SP = "pooled tset-plain"
MATCH = lambda d: f"matched tset-plain {d}"
C = lambda tag, side: f"{side} tset-plain -{tag}"   # tag in {'prob','prob-frst'}


def load_seeds():
    """seed -> mode -> cohort -> case -> (tp, fp, fn)."""
    seeds = {}
    for path in sorted(glob.glob(f"{O}/phase2_seed*.json")):
        data = json.load(open(path))
        s = int(data.get("seed_offset", 0))
        seeds[s] = {m: {c: {k: tuple(v) for k, v in cases.items()} for c, cases in per.items()}
                    for m, per in data["per_case"].items()}
    return {s: m for s, m in seeds.items() if s is not None}, sorted(seeds)


def arr(per_case, cohort, case_order):
    return np.array([per_case[cohort][k] for k in case_order[cohort]], float)


def f1_of(arr):
    return f1_arr(arr)


def main():
    seeds_by_seed, seeds = load_seeds()
    if not seeds:
        raise SystemExit("no phase2_seed*.json yet -- the CPU chain is still running")
    # case order = union of cases present in the control (SP) arm, stable across seeds
    ref = seeds_by_seed[seeds[0]].get(SP)
    if ref is None:
        raise SystemExit(f"control mode {SP!r} missing in seed {seeds[0]}: have {list(seeds_by_seed[seeds[0]])[:6]}")
    case_order = {c: sorted(ref[c]) for c in COHORTS}
    present = {m for s in seeds for m in seeds_by_seed[s]}
    n_seeds = len(seeds)

    # ---------- primary contrast S/P (control) vs S/S, per dataset + all ----------
    primary = {}
    boots = []
    if SS in present:
        per = {c: np.array([arr(seeds_by_seed[s][SS], c, case_order) for s in seeds]).sum(axis=0) / n_seeds
               for c in COHORTS}
        ctrl = {c: np.array([arr(seeds_by_seed[s][SP], c, case_order) for s in seeds]).sum(axis=0) / n_seeds
                for c in COHORTS}
        rng = np.random.RandomState(BSEED)
        idx = {c: np.arange(len(case_order[c])) for c in COHORTS}
        boot_all = np.zeros(DRAWS); boot_c = {c: np.zeros(DRAWS) for c in COHORTS}
        for d in range(DRAWS):
            ix = {c: rng.choice(idx[c], size=len(idx[c])) for c in COHORTS}
            a = np.concatenate([per[c][ix[c]] for c in COHORTS]); cc = np.concatenate([ctrl[c][ix[c]] for c in COHORTS])
            boot_all[d] = f1_of(a) - f1_of(cc)
            for c in COHORTS:
                boot_c[c][d] = f1_of(per[c][ix[c]]) - f1_of(ctrl[c][ix[c]])
        delta_all = f1_of(np.concatenate([per[c] for c in COHORTS])) - f1_of(np.concatenate([ctrl[c] for c in COHORTS]))
        dc = {c: f1_of(per[c]) - f1_of(ctrl[c]) for c in COHORTS}
        ci = {c: [float(np.percentile(boot_c[c], 2.5)), float(np.percentile(boot_c[c], 97.5))] for c in COHORTS}
        se = float(np.std(boot_all, ddof=1)) or 1e-9
        crit = float(np.quantile(boot_all, 1 - FAMILY_LEVEL))
        primary = dict(
            control=SP, arm=SS, seeds=seeds,
            delta_all=round(delta_all, 4), delta_cohorts={c: round(dc[c], 4) for c in COHORTS},
            ci95_cohorts={c: [round(ci[c][0], 4), round(ci[c][1], 4)] for c in COHORTS},
            ci95_all=[round(float(np.percentile(boot_all, 2.5)), 4), round(float(np.percentile(boot_all, 97.5)), 4)],
            one_sided_95_lower_all=round(float(np.percentile(boot_all, 5)), 4),
            simultaneous_lower=round(delta_all - crit * se, 4),
            negative_transfer=[c for c in COHORTS if ci[c][1] < NEG_TRANSFER],
        )
        primary["superiority"] = bool(primary["simultaneous_lower"] > 0 and delta_all >= DELTA_MIN)
        primary["non_inferiority"] = bool(primary["one_sided_95_lower_all"] > NI_MARGIN
                                          and not any(ci[c][1] < 0 for c in COHORTS))
        boots = boot_all

    # ---------- per-arm vs control (for the pooling figure) ----------
    def arm_vs_control(mode, cohort_only=None):
        """delta F1 of `mode` vs SP control; cohort_only -> evaluate on that one dataset only."""
        if mode not in present:
            return None
        cs = COHORTS if cohort_only is None else [cohort_only]
        per = {c: np.array([arr(seeds_by_seed[s][mode], c, case_order) for s in seeds]).sum(axis=0) / n_seeds
               for c in cs if c in seeds_by_seed[seeds[0]][mode]}
        ctrl = {c: np.array([arr(seeds_by_seed[s][SP], c, case_order) for s in seeds]).sum(axis=0) / n_seeds
                for c in cs}
        rng = np.random.RandomState(BSEED)
        idx = {c: np.arange(len(case_order[c])) for c in cs}
        boot_c = {c: np.zeros(DRAWS) for c in cs}
        for d in range(DRAWS):
            for c in cs:
                ix = rng.choice(idx[c], size=len(idx[c]))
                boot_c[c][d] = f1_of(per[c][ix]) - f1_of(ctrl[c][ix])
        dc = {c: f1_of(per[c]) - f1_of(ctrl[c]) for c in cs}
        ci = {c: [float(np.percentile(boot_c[c], 2.5)), float(np.percentile(boot_c[c], 97.5))] for c in cs}
        return {c: dict(delta=round(dc[c], 4), ci95=[round(ci[c][0], 4), round(ci[c][1], 4)]) for c in cs}

    # M-match: each dataset's matched arm is evaluated on that dataset only
    mmatch = {}
    for c in COHORTS:
        r = arm_vs_control(MATCH(c), cohort_only=c)
        if r:
            mmatch[c] = r[c]

    # mechanism arms: mixed-minus-separate gain per dataset, per channel set.
    # FIX (wk2b 05.11.2026, FEHLERBUCH 04.11.2026 06:10): the old code used
    # arm_vs_control(pool, c) = F1(pool on c) - F1(SP on c) for all three channel sets.
    # For C-all this meant pool=SP so delta was exactly 0; and C-prob and C-pf both
    # subtracted the same SP control, giving identical values. Now each channel set
    # is compared against its OWN arm (pooled - own, same channel set), so:
    #   C-prob = F1(pooled -prob on D)   - F1(own -prob on D)
    #   C-pf   = F1(pooled -prob-frst on D) - F1(own -prob-frst on D)
    #   C-all  = F1(SP on D)              - F1(SS on D)
    def arm_vs_arm(mode_a, mode_b, cohort_only=None):
        """delta F1 of mode_a vs mode_b; cohort_only -> evaluate on that one dataset only."""
        if mode_a not in present or mode_b not in present:
            return None
        cs = COHORTS if cohort_only is None else [cohort_only]
        per_a = {c: np.array([arr(seeds_by_seed[s][mode_a], c, case_order) for s in seeds]).sum(axis=0) / n_seeds
                 for c in cs if c in seeds_by_seed[seeds[0]][mode_a]}
        per_b = {c: np.array([arr(seeds_by_seed[s][mode_b], c, case_order) for s in seeds]).sum(axis=0) / n_seeds
                 for c in cs if c in seeds_by_seed[seeds[0]][mode_b]}
        cs_ok = [c for c in cs if c in per_a and c in per_b]
        rng = np.random.RandomState(BSEED)
        idx = {c: np.arange(len(case_order[c])) for c in cs_ok}
        boot_c = {c: np.zeros(DRAWS) for c in cs_ok}
        for d in range(DRAWS):
            for c in cs_ok:
                ix = rng.choice(idx[c], size=len(idx[c]))
                boot_c[c][d] = f1_of(per_a[c][ix]) - f1_of(per_b[c][ix])
        dc = {c: f1_of(per_a[c]) - f1_of(per_b[c]) for c in cs_ok}
        ci = {c: [float(np.percentile(boot_c[c], 2.5)), float(np.percentile(boot_c[c], 97.5))] for c in cs_ok}
        return {c: dict(delta=round(dc[c], 4), ci95=[round(ci[c][0], 4), round(ci[c][1], 4)]) for c in cs_ok}

    channels = {}
    for tag, label in (("prob", "C-prob"), ("prob-frst", "C-pf"), (None, "C-all")):
        if tag is None:
            own, pool = SS, SP
        else:
            own, pool = C(tag, "own"), C(tag, "pooled")
        gain = {}
        for c in COHORTS:
            g = arm_vs_arm(pool, own, cohort_only=c)   # pooled - own, same channel set
            if g and c in g:
                gain[c] = g[c]
        channels[label] = gain

    # ---------- phase2_summary.json ----------
    summary = dict(
        title="wk2 CMB Phase II -- pooling factorial, stage 2 (tset-plain), 5-fold CV, rules v2",
        control=SP, n_seeds=n_seeds, seeds=seeds,
        cohorts=COHORTS,
        pre_registered=dict(
            primary="S/P (pooled stage 2) vs S/S (separate stage 2), over all datasets",
            superiority="simultaneous lower bound > 0 AND delta(all) >= +0.01 (rules v2)",
            non_inferiority="one-sided 95 % lower bound of delta(all) > -0.02 AND no dataset 95 % interval entirely below -0.02",
        ),
        primary_contrast=primary,
        size_matched=dict(
            m_match_vs_SP=mmatch,
            diversity_effect=dict(), size_effect=dict(),
        ),
        mechanism=channels,
        stage1_unified=dict(status="pending (GPU queue: unified stage-1 U-Nets not yet run; "
                                     "candidates will use the same -z40 budget)",
                            arms=["P/S", "P/P", "PT-FT"]),
        notes=["CPU stage-2 factorial (tset-plain); stage-1 unified arms pending GPU queue.",
               "F1 is lesion-level (TP = reference lesion touched by an accepted candidate).",
               "Negative transfer = a dataset whose 95 % interval lies entirely below 0."],
    )
    # size/diversity contrasts computed from ABSOLUTE per-dataset F1 (avoids sign confusion):
    #   diversity effect = F1(M-match on D) - F1(S/S on D)   (same size, other composition)
    #   size effect      = F1(S/P on D)   - F1(M-match on D) (same composition, more data)
    def abs_f1(mode, c):
        if mode not in present:
            return None
        pc = seeds_by_seed[seeds[0]][mode]
        if c not in pc:
            return None
        a = np.array([arr(seeds_by_seed[s][mode], c, case_order) for s in seeds]).sum(axis=0) / n_seeds
        return f1_of(a)

    size = {}
    for c in COHORTS:
        f_ss = abs_f1(SS, c); f_sp = abs_f1(SP, c); f_m = abs_f1(MATCH(c), c)
        if f_ss is not None and f_sp is not None and f_m is not None:
            size[c] = dict(F1_SS=round(f_ss, 4), F1_SP=round(f_sp, 4), F1_Mmatch=round(f_m, 4),
                           diversity_effect=round(f_m - f_ss, 4), size_effect=round(f_sp - f_m, 4))
    summary["size_matched"]["size_and_diversity_per_dataset"] = size

    os.makedirs(MB_PAPER, exist_ok=True)
    with open(f"{MB_PAPER}/phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {MB_PAPER}/phase2_summary.json (seeds={seeds})")
    print(f"primary: delta_all={primary.get('delta_all')} sup={primary.get('superiority')} ni={primary.get('non_inferiority')}")

    # ---------- phase2_channels_summary.json (mechanism: mixed-minus-separate per channel set) ----------
    channels_summary = dict(
        title="wk2 CMB Phase II -- mechanism arms: mixed-minus-separate gain per dataset and patch-channel set",
        control=SP, n_seeds=n_seeds, seeds=seeds, cohorts=COHORTS,
        channel_sets={"C-prob": "stage-1 probability only (own/pooled tset-plain -prob)",
                      "C-pf": "FRST + probability (own/pooled tset-plain -prob-frst)",
                      "C-all": "image + FRST + probability (= S/S and S/P above)"},
        definition="gain[D, channel set] = F1(pooled stage 2 on D) - F1(own stage 2 on D), delta F1 with 95 % CI "
                   "from a case-level bootstrap; positive = mixing helps on D with that channel set",
        mechanism=channels,
        note="Question: does mixing help more when the classifier sees only the scanner-independent channels?",
    )
    with open(f"{MB_PAPER}/phase2_channels_summary.json", "w") as f:
        json.dump(channels_summary, f, indent=1)
    print(f"wrote {MB_PAPER}/phase2_channels_summary.json")

    # ---------- phase2_size_summary.json (size-matched control: more data vs more diverse data) ----------
    size_summary = dict(
        title="wk2 CMB Phase II -- size-matched control: is it more data or more diverse data?",
        control=SP, n_seeds=n_seeds, seeds=seeds, cohorts=COHORTS,
        m_match="mixed classifier trained on EXACTLY as many training candidates as the dataset's separate "
                "classifier sees (same fold, same positive/negative counts), drawn from the pooled candidates of "
                "all datasets, evaluated on that dataset",
        contrasts={"diversity_effect": "F1(M-match on D) - F1(S/S on D)  (same size, other composition)",
                   "size_effect": "F1(S/P on D) - F1(M-match on D)  (same composition, more data)"},
        m_match_vs_SP=mmatch,
        per_dataset=size,
        note="M-match arm evaluated on its own dataset only (matched <model> <D> scores are per-dataset).",
    )
    with open(f"{MB_PAPER}/phase2_size_summary.json", "w") as f:
        json.dump(size_summary, f, indent=1)
    print(f"wrote {MB_PAPER}/phase2_size_summary.json")

    # ---------- figures ----------
    make_fig2_pooling(summary)
    make_fig2b_channels(channels)
    make_fig2c_size(size)
    return 0


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150,
                         "axes.grid": True, "grid.alpha": 0.3})
    return plt


def make_fig2_pooling(summary):
    plt = _style()
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cohorts = COHORTS
    arms = []   # (label, {c:(delta, ci)})
    prim = summary.get("primary_contrast", {})
    if prim.get("delta_cohorts"):
        arms.append(("S/S (separate stage 2)", {c: (prim["delta_cohorts"][c], prim["ci95_cohorts"][c]) for c in cohorts}))
    mm = summary.get("size_matched", {}).get("m_match_vs_SP", {})
    if mm:
        arms.append(("M-match (size-matched mixed)", {c: (mm[c]["delta"], mm[c]["ci95"]) for c in cohorts if c in mm}))
    x = np.arange(len(cohorts))
    n = max(len(arms), 1)
    w = 0.8 / n
    colors = ["#2b6ca3", "#d9822b"]
    for i, (lab, per) in enumerate(arms):
        for j, c in enumerate(cohorts):
            if c not in per:
                continue
            d, ci = per[c]
            lo, hi = ci
            ax.errorbar(x[j] + (i - (n - 1) / 2) * w, d, yerr=[[d - lo], [hi - d]],
                        fmt="o", color=colors[i % len(colors)], capsize=4, ms=6,
                        label=lab if j == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cohorts)
    ax.set_ylabel("Delta F1 vs S/P control")
    ax.set_title("Phase II pooling factorial -- delta F1 per dataset (95 % CI), stage 2 tset-plain")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = f"{P}/fig2_pooling.png"; fig.savefig(out); plt.close(fig)
    print("wrote", out)


def make_fig2b_channels(channels):
    plt = _style()
    import numpy as np
    order = ["C-prob", "C-pf", "C-all"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cohorts = COHORTS
    x = np.arange(len(order))
    w = 0.24
    cols = ["#2b6ca3", "#5aa469", "#c0392b"]
    for j, c in enumerate(cohorts):
        vals = [channels.get(o, {}).get(c, {}).get("delta", 0.0) for o in order]
        cis = [channels.get(o, {}).get(c, {}).get("ci95", [0, 0]) for o in order]
        ax.errorbar(x + (j - 1) * w, vals, yerr=[[v - lo for v, (lo, hi) in zip(vals, cis)],
                                                   [hi - v for v, (lo, hi) in zip(vals, cis)]],
                    fmt="o", color=cols[j], capsize=3, ms=5, label=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{o}\n({c})" for o, c in zip(order, ("probability only", "FRST + probability", "image + FRST + probability"))], fontsize=8)
    ax.set_ylabel("Mixed minus separate (delta F1)")
    ax.set_title("Mechanism: does mixing help more with scanner-independent channels?")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = f"{P}/fig2b_channels.png"; fig.savefig(out); plt.close(fig)
    print("wrote", out)


def make_fig2c_size(size):
    plt = _style()
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cohorts = [c for c in COHORTS if c in size]
    x = np.arange(len(cohorts)); w = 0.36
    div = [size[c]["diversity_effect"] for c in cohorts]
    sze = [size[c]["size_effect"] for c in cohorts]
    ax.bar(x - w / 2, div, w, color="#5aa469", label="diversity effect (M-match - S/S)")
    ax.bar(x + w / 2, sze, w, color="#2b6ca3", label="size effect (S/P - M-match)")
    ax.axhline(0, color="k", lw=0.8)
    for i, c in enumerate(cohorts):
        ax.text(x[i] - w / 2, div[i] + (0.003 if div[i] >= 0 else -0.008), f"{div[i]:+.3f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, sze[i] + (0.003 if sze[i] >= 0 else -0.008), f"{sze[i]:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cohorts)
    ax.set_ylabel("Delta F1 (lesion-level)")
    ax.set_title("Size vs diversity: more data or more diverse data?")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = f"{P}/fig2c_size_vs_diversity.png"; fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
