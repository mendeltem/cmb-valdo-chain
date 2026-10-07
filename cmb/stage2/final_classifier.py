"""The deliverable second stage: ONE 3D-CNN candidate classifier trained on the candidates of ALL cohorts, saved to disk.

What the experiments established (WORKLOG 5d, 5e, 5j, 5z; OVERVIEW-tests): per cohort the candidate sets were too small
to learn from, the classifier trained on the POOL of all cohorts is the one that works (SWI +0.045 [+0.012; +0.085] against
the hard CSF rule, VALDO and T2* equal to it). Those measurements were nested cross-validations: every candidate was
scored by a model that had not seen its case, and no model was kept. The product needs weights. This module

  1. pools the candidate sets (image, FRST and first-stage probability cubes around every component above the low
     threshold 0.15 / 1 mm3, CSF kept) of all given cohorts,
  2. picks the decision threshold honestly: 5-fold cross-validation BY CASE over the pool (out-of-fold scores of every
     candidate, three members per fold), pooled lesion F1 over the threshold grid, one threshold for the product,
     plus the per-cohort scores at that threshold for the record,
  3. trains the final members (default 3 seeds) on ALL candidates and writes ``classifier.pt`` with the weights, the
     threshold, the crop and the channel order, and ``final_classifier.json`` with every number of step 2.

Architecture, optimiser, augmentation and epochs are those of ``cmb.stage2.classifier.fit_predict_cnn`` (copied, not
imported: that function returns scores, not networks). ``cmb.stage2.classifier.fit_predict_cnn`` itself is used for the
out-of-fold scores, so step 2 is the published procedure.

    python -m cmb.stage2.final_classifier --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich \
        t2star=<MB>/stage2/a03-aniso-e60-cat-t2s-weich swi=<MB>/stage2/a03-aniso-e60-cat-swi-weich \
        --out ergebnisse/stage2/final_classifier
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from cmb.stage2.apply import load, per_case, pooled_f1
from cmb.stage2.classifier import CROP, JITTER, THRESHOLDS, fit_predict_cnn
from cmb.stage2.pooled import best_threshold, scores_of

ROOT = "/home/uchralt/data/work/valdo-t2s"
CSF_LABELS = (4, 5, 14, 15, 24, 43, 44)


def build_net(nn):
    def block(c_in, c_out):
        return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True),
                             nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True))
    return nn.Sequential(block(3, 16), nn.MaxPool3d(2), block(16, 32), nn.MaxPool3d(2), block(32, 64),
                         nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))


def train_members(train_x, train_y, seeds, epochs):
    """Identical to the training loop of fit_predict_cnn (same seeds, batches, jitter, flips), but returns the networks."""
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = [(s - c) // 2 for s, c in zip(train_x.shape[-3:], CROP)]
    nets = []
    for member in range(seeds):
        torch.manual_seed(member); rng = np.random.RandomState(member)
        net = build_net(nn).to(device)
        optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        positive_weight = torch.tensor([(train_y == 0).sum() / max((train_y == 1).sum(), 1)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        target = train_y.astype(np.float32)
        for _ in range(epochs):
            net.train()
            order = rng.permutation(len(train_x))
            for start in range(0, len(order), 16):
                idx = order[start:start + 16]
                batch = []
                for i in idx:
                    z0, y0, x0 = (b - j + rng.randint(0, 2 * j + 1) for b, j in zip(base, JITTER))
                    p = train_x[i][:, z0:z0 + CROP[0], y0:y0 + CROP[1], x0:x0 + CROP[2]]
                    for axis in (1, 2, 3):
                        if rng.rand() < 0.5:
                            p = np.flip(p, axis)
                    batch.append(np.ascontiguousarray(p, dtype=np.float32))
                x = torch.from_numpy(np.stack(batch)).to(device)
                y = torch.from_numpy(target[idx]).to(device)
                optimiser.zero_grad(set_to_none=True)
                loss_fn(net(x).squeeze(1), y).backward()
                optimiser.step()
            schedule.step()
        net.eval()
        nets.append(net.cpu())
        print(f"  member {member + 1}/{seeds} trained on {len(train_x)} candidates", flush=True)
    return nets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", nargs="+", required=True, help="NAME=FOLDER with candidates.json and patches.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5, help="folds BY CASE for the threshold choice")
    ap.add_argument("--seeds", type=int, default=3, help="members of the final classifier")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else f"{ROOT}/{a.out}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    xs, ys, rows, cohort_of, n_ref, n_cases, per_cohort = [], [], [], [], 0, 0, {}
    for spec in a.cohort:
        name, folder = spec.split("=", 1)
        info, patches, c_rows, y, _, c_ref, c_cases = load(folder)
        for r in c_rows:
            r = dict(r); r["cohort"] = name; r["case"] = f"{name}:{r['case']}"; rows.append(r)
        xs.append(np.asarray(patches)); ys.append(y); cohort_of += [name] * len(c_rows)
        n_ref += c_ref; n_cases += c_cases
        per_cohort[name] = dict(folder=folder, candidates=len(c_rows), positives=int(y.sum()), cases=c_cases, reference=c_ref,
                                case_info={f"{name}:{c}": v for c, v in info["cases"].items()})
        print(f"{name}: {len(c_rows)} candidates ({int(y.sum())} positive), {c_cases} cases, {c_ref} reference lesions", flush=True)
    x = np.concatenate(xs); y = np.concatenate(ys); cohort_of = np.array(cohort_of)
    assert x.shape[1:] == (3, 20, 40, 40) or x.shape[1] == 3, x.shape
    cases_all = {}
    for pc in per_cohort.values():
        cases_all.update(pc["case_info"])

    # 2) honest threshold: 5-fold CV by case over the pool
    rng = np.random.RandomState(0)
    case_ids = sorted(cases_all)
    rng.shuffle(case_ids)
    fold_of_case = {c: i % a.folds for i, c in enumerate(case_ids)}
    fold = np.array([fold_of_case[r["case"]] for r in rows])
    oof = np.zeros(len(rows), np.float32)
    for j in range(a.folds):
        held = fold == j
        oof[held] = fit_predict_cnn(x[~held], y[~held], x[held], seed=j, epochs=a.epochs)
        print(f"fold {j}: {int(held.sum())} candidates scored out of fold  [{time.time() - t0:.0f}s]", flush=True)
    threshold = best_threshold(rows, oof, n_ref, n_cases)
    pooled = {k: round(v, 3) for k, v in scores_of(rows, oof >= threshold, n_ref, n_cases).items()}
    prob = np.array([r["probability"] for r in rows]); vox = np.array([r["voxels"] for r in rows])
    csf = np.array([r.get("tissue", -1) in CSF_LABELS for r in rows])
    fixed = (prob >= 0.3) & (vox >= 8)
    report = dict(threshold=threshold, folds=a.folds, epochs=a.epochs, candidates=len(rows), positives=int(y.sum()),
                  cases=n_cases, reference=n_ref, pooled_out_of_fold=pooled, per_cohort={})
    for name in per_cohort:
        m = cohort_of == name
        c_rows = [r for r, k in zip(rows, m) if k]
        c_ref, c_cases = per_cohort[name]["reference"], per_cohort[name]["cases"]
        own_best = best_threshold(c_rows, oof[m], c_ref, c_cases)
        res = {"probability (fixed cell)": scores_of(c_rows, fixed[m], c_ref, c_cases),
               "probability+csf": scores_of(c_rows, fixed[m] & ~csf[m], c_ref, c_cases),
               "final classifier, shared threshold": scores_of(c_rows, oof[m] >= threshold, c_ref, c_cases),
               "final classifier, own best threshold (optimistic)": scores_of(c_rows, oof[m] >= own_best, c_ref, c_cases)}
        res = {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}
        res["own_best_threshold"] = own_best
        ids = sorted(per_cohort[name]["case_info"]); cases = per_cohort[name]["case_info"]
        A = per_case(c_rows, oof[m] >= threshold, cases)
        for label, acc in (("probability (fixed cell)", fixed[m]), ("probability+csf", fixed[m] & ~csf[m])):
            B = per_case(c_rows, acc, cases)
            draws = []
            r2 = np.random.RandomState(0)
            for _ in range(5000):
                ix = r2.randint(len(ids), size=len(ids))
                draws.append(pooled_f1([A[ids[i]] for i in ix]) - pooled_f1([B[ids[i]] for i in ix]))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            res["final classifier, shared threshold"]["paired_vs_" + label] = dict(
                delta_f1=round(pooled_f1(list(A.values())) - pooled_f1(list(B.values())), 3), ci95=[round(float(lo), 3), round(float(hi), 3)])
        report["per_cohort"][name] = res
        print(f"\n{name}: " + "  ".join(f"{k}: F1 {v['f1']:.3f}" for k, v in res.items() if isinstance(v, dict) and "f1" in v), flush=True)
    print(f"\nshared threshold {threshold:.2f}: pooled out-of-fold F1 {pooled['f1']:.3f} (P {pooled['precision']:.2f} S {pooled['sensitivity']:.2f})", flush=True)

    # 3) final members on ALL candidates
    import torch
    nets = train_members(x, y, a.seeds, a.epochs)
    bundle = dict(members=[n.state_dict() for n in nets], threshold=threshold, crop=list(CROP), channels=["image", "frst", "probability"],
                  stored_patch=list(x.shape[-3:]), candidate_threshold=0.15, candidate_min_mm3=1.0, trained_on=list(per_cohort),
                  n_candidates=len(rows), n_positive=int(y.sum()), epochs=a.epochs, seeds=a.seeds, created=time.strftime("%Y-%m-%d %H:%M"))
    torch.save(bundle, f"{out}/classifier.pt")
    report.update(seeds=a.seeds, duration_s=round(time.time() - t0, 1), cohorts={k: {kk: vv for kk, vv in v.items() if kk != "case_info"} for k, v in per_cohort.items()},
                  classifier=f"{out}/classifier.pt")
    json.dump(report, open(f"{out}/final_classifier.json", "w"), indent=1)
    print(f"FINISHED {out}/classifier.pt ({a.seeds} members, threshold {threshold:.2f}) in {report['duration_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
