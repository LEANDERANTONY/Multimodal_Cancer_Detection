"""In-distribution confound tax with a patient-level bootstrap CI, from a per-case CV score CSV
(columns fold, case, scanner, y, p_max, p_sum, cc_psz — written by tight_battery.py / loose_cv_reinfer.py).

tax = mean per-fold MIXED-scanner AUROC - mean over scanners of the per-fold-averaged WITHIN-scanner
AUROC (cells need >=8 pos and >=8 neg). Same definition as cv_detection.py / tight_battery.py.
Usage: python tools/confound_tax_ci.py <cv_scores.csv> [n_boot]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCANNERS = ["Siemens", "Toshiba", "Philips"]


def tax(d, s):
    mixed = np.mean([roc_auc_score(g.y, g[s]) for _, g in d.groupby("fold") if g.y.nunique() == 2])
    within = []
    for nm in SCANNERS:
        cells = [roc_auc_score(g.y, g[s]) for _, g in d[d.scanner == nm].groupby("fold")
                 if g.y.sum() >= 8 and (1 - g.y).sum() >= 8]
        within.append(np.mean(cells))
    return mixed - np.mean(within), mixed, within


def main():
    d = pd.read_csv(sys.argv[1]); n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    d["pat"] = d.case.str.split("_").str[0]
    pats = d.pat.unique(); idx = d.groupby("pat").indices; rng = np.random.default_rng(0)
    print(f"{sys.argv[1]}: n={len(d)} pos={int(d.y.sum())} folds={sorted(d.fold.unique())}")
    for s in ["p_max", "p_sum", "cc_psz"]:
        t0, m, w = tax(d, s); bs = []
        for _ in range(n_boot):
            b = d.iloc[np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])]
            try: bs.append(tax(b, s)[0])
            except ValueError: pass
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {s:7s} tax={t0:+.3f} 95%CI [{lo:+.3f},{hi:+.3f}]  mixed={m:.3f}  within "
              + " ".join(f"{n}={v:.3f}" for n, v in zip(SCANNERS, w)) + f"  mixed>all-within={m > max(w)}")


if __name__ == "__main__":
    main()
