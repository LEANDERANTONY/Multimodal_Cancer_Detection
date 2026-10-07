"""Deployment gap: oracle-mask crops vs predicted-mask crops, paired on the same external patients.

MSD (both classes): detection AUROC for p_max / cc_psz and lesion Dice on PDAC cases, each as
oracle, predicted and the paired difference (oracle - predicted) with a patient-level bootstrap CI.
NIH (all healthy): specificity at the pooled Dutch-CV Youden threshold (external/threshold_check.csv).
MSD and NIH are never pooled.

    python tools/deploy_gap.py [arm ...]        # default: baseline_oof totalseg
Writes reports/deployment_roi/deploy_gap_external.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ORACLE = "reports/nnunet_summaries/external/tight_external.csv"
THR = "reports/nnunet_summaries/external/threshold_check.csv"
OUT = "reports/deployment_roi/deploy_gap_external.csv"
SCORES = ["p_max", "cc_psz"]
N_BOOT = 2000


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"gap_lo": lo, "gap_hi": hi}


def main():
    arms = sys.argv[1:] or ["baseline_oof", "totalseg"]
    rng = np.random.default_rng(0)
    oracle = pd.read_csv(ORACLE)
    thr = pd.read_csv(THR)
    thr = thr[(thr.model.str.startswith("ensemble")) & (thr.op == "youden")].set_index("score").thr
    rows = []
    for arm in arms:
        pred = pd.read_csv(f"reports/deployment_roi/detect_{arm}_external.csv")
        m = oracle.merge(pred, on=["case", "source", "y"], suffixes=("_o", "_p"))
        msd, nih = m[m.source == "MSD"].reset_index(drop=True), m[m.source == "NIH"]
        y = msd.y.values
        idx = [rng.integers(0, len(msd), len(msd)) for _ in range(N_BOOT)]
        idx = [i for i in idx if 0 < y[i].sum() < len(i)]
        for s in SCORES:
            o, p = msd[f"{s}_o"].values, msd[f"{s}_p"].values
            ao, ap = roc_auc_score(y, o), roc_auc_score(y, p)
            d = [roc_auc_score(y[i], o[i]) - roc_auc_score(y[i], p[i]) for i in idx]
            rows.append({"arm": arm, "metric": f"MSD AUROC {s}", "n": len(msd), "oracle": ao, "predicted": ap,
                         "gap": ao - ap, **ci(d)})
        pos = msd[msd.y == 1].reset_index(drop=True)
        do, dp = pos.dice_o.fillna(0).values, pos.dice_p.fillna(0).values
        d = [(do[i] - dp[i]).mean() for i in (rng.integers(0, len(pos), len(pos)) for _ in range(N_BOOT))]
        rows.append({"arm": arm, "metric": "MSD lesion Dice", "n": len(pos), "oracle": do.mean(),
                     "predicted": dp.mean(), "gap": do.mean() - dp.mean(), **ci(d)})
        for s in SCORES:
            so, sp = (nih[f"{s}_o"] < thr[s]).mean(), (nih[f"{s}_p"] < thr[s]).mean()
            rows.append({"arm": arm, "metric": f"NIH specificity {s} @Dutch Youden", "n": len(nih),
                         "oracle": so, "predicted": sp, "gap": so - sp})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
