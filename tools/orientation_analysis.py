"""What the inverted cranio-caudal headers (tools/orientation_check.py) explain.

1. Segmenter failures: empty pancreas masks per TotalSegmentator setting (as published, robust_crop, + neighbouring
   organs) by orientation, over all 1,964 Dutch scans.
2. Is inversion itself a confound? Inverted share by scanner and label; Fisher test and odds ratio PDAC vs inverted.
3. Do the detectors use it? On the patient-disjoint evaluation set: AUROC on correctly oriented scans only vs all, and
   AUROC among the inverted scans, for the 3D model (cc_psz) and the 2.5D ERM classifier (seed-averaged).

    python tools/orientation_analysis.py
Writes reports/deployment_roi/orientation_summary.csv and prints the tables.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cohort import eval_set  # noqa: E402
from final_stats import single_ci  # noqa: E402

ARMS = {"totalseg": "roi_subset pancreas", "totalseg_rc": "+ robust_crop", "totalseg_nb": "+ neighbouring organs"}
OUT = "reports/deployment_roi/orientation_summary.csv"


def main():
    o = pd.read_csv("reports/deployment_roi/orientation_check.csv")
    cv = pd.read_csv("reports/nnunet_summaries/tight_battery/cv_scores.csv")
    d = cv[["case", "scanner", "y", "cc_psz"]].merge(o[["case", "orientation"]], on="case")
    assert len(d) == 1964, len(d)
    rows = []
    print(pd.crosstab([d.scanner, d.y], d.orientation, margins=True))

    for arm, name in ARMS.items():
        sk = f"data/processed/ct/deploy_crops/{arm}_dutch/skipped.txt"
        failed = set(open(sk).read().split()) if os.path.exists(sk) else set()
        d[f"fail_{arm}"] = d.case.isin(failed)
        for state, g in d.groupby("orientation"):
            rows.append({"block": "segmenter_failures", "arm": name, "orientation": state, "n": len(g),
                         "failed": int(g[f"fail_{arm}"].sum()), "failed_pct": 100 * g[f"fail_{arm}"].mean()})
        print(f"\n{name}: failures by orientation\n", pd.crosstab(d.orientation, d[f"fail_{arm}"], margins=True))

    inv = d.orientation == "inverted"
    for (s, y), g in d.groupby(["scanner", "y"]):
        rows.append({"block": "inverted_share", "scanner": s, "y": y, "n": len(g),
                     "inverted": int((g.orientation == "inverted").sum()),
                     "inverted_pct": 100 * (g.orientation == "inverted").mean()})
    tab = pd.crosstab(inv, d.y)
    odds, p = fisher_exact(tab.values)
    rows.append({"block": "inverted_vs_pdac", "n": len(d), "inverted": int(inv.sum()),
                 "pdac_pct_inverted": 100 * d.y[inv].mean(), "pdac_pct_other": 100 * d.y[~inv].mean(),
                 "odds_ratio": odds, "fisher_p": p, "auroc_inverted_only": roc_auc_score(d.y, inv.astype(int))})
    print(f"\ninverted {int(inv.sum())}: PDAC {100 * d.y[inv].mean():.1f} % vs {100 * d.y[~inv].mean():.1f} %; "
          f"OR {odds:.2f}, Fisher p {p:.2g}")
    for s, g in d.groupby("scanner"):
        gi = g.orientation == "inverted"
        if gi.sum() and g.y.nunique() == 2:
            print(f"  {s}: inverted {int(gi.sum())}/{len(g)}, PDAC among inverted {100 * g.y[gi].mean():.0f} % "
                  f"vs {100 * g.y[~gi].mean():.0f} %")

    p25 = pd.concat(pd.read_csv(f"reports/panel/erm_resnet18_d20_s{s}/test_cv{k}.csv") for s in range(3) for k in range(5))
    p25 = p25.groupby("case", as_index=False).p.mean().rename(columns={"p": "p_2p5d"})
    e = eval_set(d.merge(p25, on="case"))
    for col, model in (("cc_psz", "3D nnU-Net (cc_psz)"), ("p_2p5d", "2.5D ERM")):
        for sub, g in (("all", e), ("header_ok", e[e.orientation == "header_ok"]),
                       ("not_inverted", e[e.orientation != "inverted"]), ("inverted", e[e.orientation == "inverted"])):
            if g.y.nunique() < 2:
                continue
            a, lo, hi = single_ci(g.y.values, g[col].values)
            rows.append({"block": "detector_auroc", "model": model, "subset": sub, "n": len(g), "pos": int(g.y.sum()),
                         "auroc": a, "lo": lo, "hi": hi})
            print(f"{model:22s} {sub:12s} n={len(g):4d} pos={int(g.y.sum()):3d} AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(OUT)


if __name__ == "__main__":
    main()
