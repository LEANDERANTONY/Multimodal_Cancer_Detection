"""Mitigation panel: evaluate the per-case scores of tools/panel_train.py, per arm, seed by seed.

Per run (arm x seed), when its folds are complete:
    cv:   mixed AUROC, within-scanner AUROCs and the confound tax with a patient bootstrap CI
          (same definition as tools/confound_tax_ci.py, on the out-of-fold test scores)
    lomo: AUROC on each held-out manufacturer (model never saw that scanner)
    ext:  MSD/NIH AUROC of the mean score over the 5 CV fold models
Then mean +- SD over the seeds per arm.

    python tools/panel_eval.py [n_boot]
Writes reports/panel/panel_results.csv (per arm x seed) and panel_summary.csv (per arm).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confound_tax_ci import SCANNERS, tax  # noqa: E402

REPORTS = "reports/panel"


def boot_ci(d, n_boot, seed=0):
    d = d.assign(pat=d.case.str.split("_").str[0])
    pats, idx, rng = d.pat.unique(), d.groupby("pat").indices, np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        b = d.iloc[np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])]
        try:
            bs.append(tax(b, "p")[0])
        except ValueError:
            pass
    return np.percentile(bs, [2.5, 97.5])


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rows = []
    for rdir in sorted(glob.glob(f"{REPORTS}/*_s[0-9]")):
        run = os.path.basename(rdir)
        if "tune" in run or "timing" in run:
            continue
        arm, seed = run.split("_")[0], int(run.rsplit("_s", 1)[1])
        r = {"run": run, "arm": arm, "seed": seed}
        cv = [f"{rdir}/test_cv{k}.csv" for k in range(5)]
        if all(map(os.path.exists, cv)):
            d = pd.concat(map(pd.read_csv, cv))
            t, mixed, within = tax(d, "p")
            lo, hi = boot_ci(d, n_boot)
            r.update(cv_auc_pooled=roc_auc_score(d.y, d.p), cv_mixed=mixed, cv_tax=t, cv_tax_lo=lo, cv_tax_hi=hi,
                     **{f"cv_within_{s}": w for s, w in zip(SCANNERS, within)})
            ext = pd.concat(pd.read_csv(f"{rdir}/ext_cv{k}.csv") for k in range(5))
            ext = ext.groupby(["case", "scanner", "y"], as_index=False).p.mean()
            r["ext_auc"] = roc_auc_score(ext.y, ext.p)
            msd = ext[ext.scanner == "MSD"]
            if msd.y.nunique() == 2:
                r["ext_msd_auc"] = roc_auc_score(msd.y, msd.p)
        for k, s in enumerate(SCANNERS):
            f = f"{rdir}/test_lomo{k}.csv"
            if os.path.exists(f):
                d = pd.read_csv(f)
                r[f"lomo_{s}"] = roc_auc_score(d.y, d.p)
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS}/panel_results.csv", index=False)
    num = res.drop(columns=["run", "seed"]).groupby("arm")
    summ = num.mean().round(3).add_suffix("_mean").join(num.std().round(3).add_suffix("_sd"))
    summ.insert(0, "n_seeds", res.groupby("arm").size())
    summ = summ[["n_seeds"] + sorted(c for c in summ.columns if c != "n_seeds")]
    summ.to_csv(f"{REPORTS}/panel_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
