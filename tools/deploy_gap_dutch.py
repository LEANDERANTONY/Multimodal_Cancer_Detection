"""Deployment gap on the Dutch cohort: oracle-mask crops (tight-CV cv_scores.csv) vs predicted-mask crops
(tools/deploy_infer_oof.py), paired on the same cases, same out-of-fold models.

Per arm and score: mixed-scanner AUROC, within-scanner AUROC per scanner and the confound tax (same
definition as tools/confound_tax_ci.py), each as oracle and predicted, plus predicted - oracle with a
patient-level bootstrap CI. Two populations:
  paired - cases with a usable predicted mask (complete case);
  itt    - every case the arm was run on; a segmenter failure (empty / missing mask) gets the lowest
           observed score, i.e. it is called negative, as a deployed pipeline would.
A failure = a case stage 2 skipped for lack of a usable mask (deploy_crops/<arm>_dutch/skipped.txt). Failure counts
are over all 1,964 scans; detection metrics over the patient-disjoint evaluation set (tools/eval_cohort.py).

    python tools/deploy_gap_dutch.py [arm ...]      # default: baseline_oof totalseg
Writes reports/deployment_roi/deploy_gap_dutch.csv and deploy_fail_dutch.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "tools")
from confound_tax_ci import SCANNERS, tax  # noqa: E402
from eval_cohort import eval_set  # noqa: E402

ORACLE = "reports/nnunet_summaries/tight_battery/cv_scores.csv"
SCORES = ["p_max", "p_sum", "cc_psz"]
N_BOOT = 1000


def stats(d, s):
    t, mixed, within = tax(d, s)
    return [t, mixed] + list(within)


def main():
    arms = sys.argv[1:] or ["baseline_oof", "totalseg"]
    oracle = pd.read_csv(ORACLE)
    rng = np.random.default_rng(0)
    rows, fails = [], []
    names = ["tax", "mixed"] + [f"within_{n}" for n in SCANNERS]
    for arm in arms:
        pred = pd.read_csv(f"reports/deployment_roi/detect_{arm}_dutch.csv")
        sk = f"data/processed/ct/deploy_crops/{arm}_dutch/skipped.txt"
        ran = set(pred.case) | (set(open(sk).read().split()) if os.path.exists(sk) else set())
        m = oracle[oracle.case.isin(ran)].merge(pred[["case"] + SCORES], on="case", how="left",
                                                suffixes=("_o", "_p")).reset_index(drop=True)
        ok = m[f"{SCORES[0]}_p"].notna()
        for g, d in m.groupby("scanner"):
            bad = ~ok[d.index]
            fails.append({"arm": arm, "scanner": g, "n": len(d), "pos": int(d.y.sum()), "failed": int(bad.sum()),
                          "failed_pos": int((bad & (d.y == 1)).sum())})
        m = eval_set(m)  # failure counts above use every scan; detection metrics the patient-disjoint set
        ok = m[f"{SCORES[0]}_p"].notna()
        for pop in ("paired", "itt"):
            d = m[ok].copy() if pop == "paired" else m.copy()
            for s in SCORES:
                d[f"{s}_p"] = d[f"{s}_p"].fillna(d[f"{s}_p"].min())
            d = d.reset_index(drop=True)
            d["pat"] = d.case.str.split("_").str[0]
            idx, pats = d.groupby("pat").indices, d.pat.unique()
            for s in SCORES:
                do, dp = d.assign(**{s: d[f"{s}_o"]}), d.assign(**{s: d[f"{s}_p"]})
                vo, vp = stats(do, s), stats(dp, s)
                bs = []
                for _ in range(N_BOOT):
                    b = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
                    try:
                        bs.append(np.subtract(stats(dp.iloc[b], s), stats(do.iloc[b], s)))
                    except ValueError:
                        pass
                lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
                for k, nm in enumerate(names):
                    rows.append({"arm": arm, "population": pop, "n": len(d), "pos": int(d.y.sum()), "score": s,
                                 "metric": nm, "oracle": vo[k], "predicted": vp[k], "diff": vp[k] - vo[k],
                                 "diff_lo": lo[k], "diff_hi": hi[k]})
                t, mx = rows[-len(names)], rows[-len(names) + 1]
                print(f"{arm:12s} {pop:6s} n={len(d)} {s:6s} tax {t['oracle']:+.3f} -> {t['predicted']:+.3f} "
                      f"(diff {t['diff']:+.3f} [{t['diff_lo']:+.3f},{t['diff_hi']:+.3f}]); "
                      f"mixed {mx['oracle']:.3f} -> {mx['predicted']:.3f} "
                      f"(diff {mx['diff']:+.3f} [{mx['diff_lo']:+.3f},{mx['diff_hi']:+.3f}])", flush=True)
    pd.DataFrame(rows).to_csv("reports/deployment_roi/deploy_gap_dutch.csv", index=False)
    f = pd.DataFrame(fails)
    f.to_csv("reports/deployment_roi/deploy_fail_dutch.csv", index=False)
    print(f.to_string(index=False))


if __name__ == "__main__":
    main()
