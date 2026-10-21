"""Mitigation panel, planted-shortcut control: does each arm stop using a shortcut we know is there?

Per planted run (arm x seed; tools/panel_train.py --planted, tools/panel_dfr.py --planted), pooled over the 5 CV
test folds: AUROC on clean test scans (no marker), AUROC with the marker on exactly the non-PDAC scans (the
shortcut points the wrong way), and the drop between them with a patient bootstrap CI. A model that ignores the
marker has a drop near 0; one that relies on it collapses.

    python tools/panel_planted_eval.py [n_boot]
Writes reports/panel/planted_results.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPORTS = "reports/panel"


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rows = []
    for rdir in sorted(glob.glob(f"{REPORTS}/*_plant*_s[0-9]")):
        run = os.path.basename(rdir)
        files = [(f"{rdir}/test_cv{k}.csv", f"{rdir}/test_flip_cv{k}.csv") for k in range(5)]
        if not all(os.path.exists(f) for pair in files for f in pair):
            continue
        clean = pd.concat(pd.read_csv(a) for a, _ in files)
        flip = pd.concat(pd.read_csv(b) for _, b in files)
        d = clean.merge(flip[["case", "p"]], on="case", suffixes=("", "_flip"))
        d["pat"] = d.case.str.split("_").str[0]
        a_c, a_f = roc_auc_score(d.y, d.p), roc_auc_score(d.y, d.p_flip)
        idx, pats, rng = d.groupby("pat").indices, d.pat.unique(), np.random.default_rng(0)
        bs = []
        for _ in range(n_boot):
            b = d.iloc[np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])]
            bs.append(roc_auc_score(b.y, b.p) - roc_auc_score(b.y, b.p_flip))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"run": run, "arm": run.split("_")[0], "planted": run.split("_plant")[1].split("_")[0],
                     "seed": int(run.rsplit("_s", 1)[1]), "auc_clean": a_c, "auc_flip": a_f,
                     "drop": a_c - a_f, "drop_lo": lo, "drop_hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS}/planted_results.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
