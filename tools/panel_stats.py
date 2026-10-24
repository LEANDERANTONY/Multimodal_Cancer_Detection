"""Mitigation panel statistics: every arm vs ERM on the same cases, with scores averaged over the 3 seeds.

    cv:       Dutch out-of-fold (5 CV test folds pooled)  -> DeLong AUROC difference, confound-tax difference
                                                             (paired patient bootstrap), Brier, ECE
    ext:      MSD + NIH, mean over the 5 CV fold models    -> DeLong, Brier, ECE
    lomo_<s>: held-out manufacturer s                       -> DeLong
DeLong: fast paired implementation (Sun & Xu 2014). ECE: 10 equal-width bins.

    python tools/panel_stats.py [n_boot]
Writes reports/panel/panel_stats.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confound_tax_ci import SCANNERS, tax  # noqa: E402

REPORTS = "reports/panel"
ARMS = ["erm", "dfr", "gbs", "grl", "mae"]  # arms without results yet are skipped
RUN = "{arm}_resnet18_d20_s{seed}"


def midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, a, b):
    """Paired DeLong test: returns AUROC(a), AUROC(b), difference a - b, its 95% CI and two-sided p."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([a, b])
    m, n = y.sum(), (~y).sum()
    pos, neg = preds[:, y], preds[:, ~y]
    tx = np.array([midrank(p) for p in pos])
    ty = np.array([midrank(p) for p in neg])
    tz = np.array([midrank(p) for p in np.hstack([pos, neg])])
    auc = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    d = auc[0] - auc[1]
    se = np.sqrt(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    p = 2 * norm.sf(abs(d) / se) if se > 0 else 1.0
    return auc[0], auc[1], d, d - 1.96 * se, d + 1.96 * se, p


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any())


def scores(arm, kind):
    """Seed-averaged per-case scores. kind: cv | ext | lomo<k>."""
    frames = []
    for seed in range(3):
        d = f"{REPORTS}/{RUN.format(arm=arm, seed=seed)}"
        if kind == "cv":
            f = pd.concat(pd.read_csv(f"{d}/test_cv{k}.csv") for k in range(5))
        elif kind == "ext":
            f = pd.concat(pd.read_csv(f"{d}/ext_cv{k}.csv") for k in range(5))
            f = f.groupby(["case", "scanner", "y"], as_index=False).p.mean()
        else:
            f = pd.read_csv(f"{d}/test_{kind}.csv")
        frames.append(f.assign(seed=seed))
    f = pd.concat(frames)
    keys = ["case", "scanner", "y"] + (["fold"] if kind == "cv" else [])
    return f.groupby(keys, as_index=False).p.mean().sort_values("case").reset_index(drop=True)


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rows = []
    for kind in ["cv", "ext"] + [f"lomo{k}" for k in range(3)]:
        ref = scores("erm", kind)
        label = kind if not kind.startswith("lomo") else f"lomo_{SCANNERS[int(kind[-1])]}"
        for arm in ARMS:
            try:
                d = scores(arm, kind)
            except FileNotFoundError:
                continue
            assert (d.case.values == ref.case.values).all()
            y = d.y.values
            r = {"setting": label, "arm": arm, "n": len(d), "auroc": roc_auc_score(y, d.p)}
            if arm != "erm":
                _, _, diff, lo, hi, p = delong(y, d.p.values, ref.p.values)
                r.update(auroc_diff_vs_erm=diff, diff_lo=lo, diff_hi=hi, delong_p=p)
            if kind in ("cv", "ext"):
                r.update(brier=brier_score_loss(y, d.p), ece=ece(y, d.p.values))
            if kind == "cv":
                both = d.assign(p_ref=ref.p.values)
                t_arm, t_ref = tax(both, "p")[0], tax(both.assign(p=both.p_ref), "p")[0]
                r["tax"] = t_arm
                if arm != "erm":
                    both["pat"] = both.case.str.split("_").str[0]
                    idx, pats, rng = both.groupby("pat").indices, both.pat.unique(), np.random.default_rng(0)
                    bs = []
                    for _ in range(n_boot):
                        b = both.iloc[np.concatenate([idx[q] for q in rng.choice(pats, len(pats))])]
                        try:
                            bs.append(tax(b, "p")[0] - tax(b.assign(p=b.p_ref), "p")[0])
                        except ValueError:
                            pass
                    r.update(tax_diff_vs_erm=t_arm - t_ref, tax_diff_lo=np.percentile(bs, 2.5),
                             tax_diff_hi=np.percentile(bs, 97.5))
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS}/panel_stats.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
