# How the literature selects and validates models -- and what we already do (2026-09-29; sonnet subagent, 10 searches; check citations before use)

| # | Statement | Source | For us |
|---|---|---|---|
| 1 | nnU-Net: fixed heuristic instead of hyperparameter search, 5-fold CV | Isensee 2021, Nat. Methods 18 | Tournament 2 is exactly this kind: one targeted change each, no grid/Bayes |
| 2 | "nnU-Net Revisited": four pitfalls -- innovation not separated from confounder, no standard baseline, too few/similar datasets, inconsistent reporting | Isensee 2024, arXiv 2404.09556 | Baseline present (a02 DynUNet 0.458 clean; ResEnc queued), but not the nnU-Net framework with its own preprocessing -- name as a limitation; three cohorts + Momeni = four datasets |
| 3 | Metrics Reloaded: naive pooling over hierarchical data favours lesion-rich cases; average per case | Maier-Hein 2024, Nat. Methods 21 | pooled is our main metric (sub-110 carries 75% of the test set!). Bootstrap already runs per case; additionally report mean/median per case as a second main metric (subject_level_f1 provides both) |
| 4 | Instance metric: fix the localisation criterion and assignment IN ADVANCE | same, Fig. 2 | in place: 26-neighbourhood, touch rule, giant components removed (WORKLOG.md 4.3, REPLICATION 4); challenge rule (centre 5 mm) alongside |
| 5 | Data sampling is the largest source of variance, init < 50%; many repetitions needed | Bouthillier 2021, MLSys | our case bootstrap covers the sampling; seed spread is measured (VALDO 7 seeds 0.569-0.646 with the rule; second stage 10 seeds sd 0.003-0.009) |
| 6 | Seeds produce outliers, variance does not decrease with longer training | Picard 2021 | hence the rule: a single seed decides nothing below the seed maximum of the baseline (0.646); the tournament-2 threshold +0.04 lies above that |
| 7 | HPO at < 100 cases: no evidence of benefit; nnU-Net deliberately avoids it | Isensee 2021; Auto-nnU-Net 2025 | Tournament 2 serves as proof of the search, not hope |
| 8 | VALDO overview: Dice + volumetry as the challenge metric, large spread, "gap at case level" | arXiv 2208.07167 | our lesion F1 + paired bootstrap goes beyond that |
| 9 | MicrobleedNet: internal Dice 0.73, external 0.46 -- the domain shift is larger than any seed noise | PMC12091992 (2024/25) | measured: VALDO model on catalina-SWI 0.305, on Momeni 0.613; own training per cohort is the answer |
| 10 | MixMicrobleedNet: fold=all, self-test on training data | arXiv 2108.01389 | us: 5-fold CV, test set once |
| 11 | SHIVA-CMB: 450 scans, 7 acquisitions, held-out test set + external cohort (F1 0.74, S 0.67, P 0.82) | Boutinaud 2024, Sci Rep | we have the external test (Momeni) -- but only 57 cases of one acquisition; training breadth (7 acquisitions) is what we lack |
| 12 | CenSynCMB: leakage-free per fold, spread across folds + p-values, external cohort as an additional control | He 2026, arXiv 2607.05325 | congruent with our design; we report fold spread (F1 per fold in the ranking) |

## What we change as a result (proposal)
1. **Second main metric per case** (mean and median of the F1 per case, as in Metrics Reloaded) alongside the pooled F1 in EVERY table -- cheap, the tool exists. This methodically addresses the sub-110 dominance.
2. **Explicit table of the baseline's seed null distribution** (VALDO: s42 0.646, s1 0.590, s2 0.578, s3 0.603, s4 0.608, s5 0.569, s6 0.618; s7-s9 still to be evaluated): every individual seed run of the tournament is read against this distribution, not only against s42.
3. **Sample-based second seeds also for losers** (Picard): two randomly chosen tournament-2 runs get a second seed, so the +0.04 threshold does not only secure winners.
4. **nnU-Net framework as baseline** (Revisited): the actual nnU-Net v2 with its own preprocessing on VALDO, one run -- the only gap a reviewer will surely name. Effort: installation + ~6 h GPU (1000 epochs is the nnU-Net standard).
5. Name limitations in the paper: one acquisition per cohort, Momeni as the only external cohort.
