"""Fusion without paired patients: the CT (PANORAMA) and urinary biomarker cohorts share no patient, so fusion is
studied (1) as a sequential-triage simulation with the dependence between the two tests as an explicit parameter,
and (2) as the pseudo-paired fusion the thesis used, shown only as a cautionary demonstration.

Real inputs (out-of-fold, PDAC vs non-PDAC):
    biomarker  published panel + age (log LYVE1, REG1B, TFF1, creatinine; age), repeated 5-fold CV
    ct         tight 3D nnU-Net cc_psz on the 1,964 Dutch scans (primary); 2.5D ERM, seed-averaged (sensitivity)
(1) Simulation: virtual patients keep each test's real class-conditional score distribution exactly (empirical
    quantiles) and are linked by a Gaussian copula with within-class correlation rho. Reported per rho: AUROC of the
    best combined score (logistic on both, fitted on half of the simulated patients, scored on the other half) and a
    triage pathway: urine positive at the threshold giving 95 % sensitivity on the real biomarker scores, then CT
    positive at the threshold giving 90 % sensitivity on the real CT scores. Pathway sensitivity / specificity and
    the share of CTs avoided at a high-risk screening prevalence (1 %) and at the cohort prevalence (34 %).
    95 % intervals: 200 bootstrap resamples of the real score sets.
(2) Pseudo-pairing: every biomarker sample gets a random CT case with the same label (the thesis's construction),
    fusion = logistic regression on the two scores, 5-fold CV; 20 random pairings. Control: CT case drawn ignoring
    the label.

    python tools/fusion_sim.py
Writes reports/fusion/fusion_triage.csv, reports/fusion/fusion_pseudo_pairs.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from biomarker_confound import DATA, features, oof  # noqa: E402

OUT = "reports/fusion"
RHOS = [0.0, 0.25, 0.5, 0.75]
N_SIM, N_BOOT, N_PAIR = 100_000, 200, 20
PREV = {"screening_1pct": 0.01, "cohort_34pct": 0.34}


def biomarker_scores():
    d = pd.read_csv(DATA)
    d = d.assign(y=(d.diagnosis == 3).astype(int)).reset_index(drop=True)
    return oof(features(d, "panel+age"), d.y.values, (d.sample_origin + d.y.astype(str)).values), d.y.values


def ct_scores(kind):
    if kind == "ct_3d":
        d = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")
        return d.cc_psz.values, d.y.values
    f = pd.concat(pd.read_csv(f"reports/panel/erm_resnet18_d20_s{s}/test_cv{k}.csv") for s in range(3) for k in range(5))
    f = f.groupby(["case", "y"], as_index=False).p.mean()
    return f.p.values, f.y.values


def sens_threshold(s, y, sens):
    return np.quantile(s[y == 1], 1 - sens)


def simulate(b, yb, c, yc, rho, rng):
    """Virtual patients per class: (biomarker, ct) with the real marginals and Gaussian-copula correlation rho."""
    out = {}
    for cls in (0, 1):
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], N_SIM)
        u = norm.cdf(z)
        out[cls] = (np.quantile(b[yb == cls], u[:, 0]), np.quantile(c[yc == cls], u[:, 1]))
    return out


def triage_metrics(b, yb, c, yc, rho, rng):
    sim = simulate(b, yb, c, yc, rho, rng)
    tb, tc = sens_threshold(b, yb, 0.95), sens_threshold(c, yc, 0.90)
    ub = {k: v[0] > tb for k, v in sim.items()}
    path = {k: ub[k] & (v[1] > tc) for k, v in sim.items()}
    r = {"triage_sens": path[1].mean(), "triage_spec": 1 - path[0].mean(),
         "ct_alone_sens": (sim[1][1] > tc).mean(), "ct_alone_spec": 1 - (sim[0][1] > tc).mean()}
    for name, pi in PREV.items():
        r[f"ct_avoided_{name}"] = 1 - (pi * ub[1].mean() + (1 - pi) * ub[0].mean())
    x = np.vstack([np.column_stack(sim[0]), np.column_stack(sim[1])])
    y = np.r_[np.zeros(N_SIM), np.ones(N_SIM)]
    x = np.column_stack([x[:, 0], x[:, 1]])
    idx = rng.permutation(len(y))
    tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
    m = LogisticRegression(max_iter=1000).fit(x[tr], y[tr])
    r["combined_auroc"] = roc_auc_score(y[te], m.predict_proba(x[te])[:, 1])
    return r


def pseudo_pairs(b, yb, c, yc, rng):
    rows = []
    for control in (False, True):
        for i in range(N_PAIR):
            pool = {cls: np.where(yc == cls)[0] for cls in (0, 1)}
            pick = (rng.integers(len(c), size=len(b)) if control
                    else np.array([rng.choice(pool[cls]) for cls in yb]))
            x = np.column_stack([b, c[pick]])
            p = cross_val_predict(LogisticRegression(max_iter=1000), x, yb,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=i), method="predict_proba")[:, 1]
            rows.append({"pairing": "label_mismatched_control" if control else "label_matched",
                         "repeat": i, "fusion_auroc": roc_auc_score(yb, p)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    b, yb = biomarker_scores()
    rows, pairs = [], []
    for ct in ("ct_3d", "ct_2p5d_erm"):
        c, yc = ct_scores(ct)
        base = {"ct_model": ct, "biomarker_auroc": roc_auc_score(yb, b), "ct_auroc": roc_auc_score(yc, c)}
        for rho in RHOS:
            point = triage_metrics(b, yb, c, yc, rho, rng)
            boot = []
            for _ in range(N_BOOT):
                ib, ic = rng.integers(len(b), size=len(b)), rng.integers(len(c), size=len(c))
                boot.append(triage_metrics(b[ib], yb[ib], c[ic], yc[ic], rho, rng))
            boot = pd.DataFrame(boot)
            r = {**base, "rho": rho}
            for k, v in point.items():
                r[k], r[f"{k}_lo"], r[f"{k}_hi"] = v, boot[k].quantile(0.025), boot[k].quantile(0.975)
            rows.append(r)
            print(ct, rho, {k: round(v, 3) for k, v in point.items()}, flush=True)
        pairs.append(pseudo_pairs(b, yb, c, yc, rng).assign(**base))
    pd.DataFrame(rows).to_csv(f"{OUT}/fusion_triage.csv", index=False)
    pp = pd.concat(pairs)
    pp.to_csv(f"{OUT}/fusion_pseudo_pairs.csv", index=False)
    print(pp.groupby(["ct_model", "pairing"]).fusion_auroc.agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
