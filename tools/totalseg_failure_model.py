"""Which scan features predict a TotalSegmentator failure (empty pancreas), with scanner and cancer status
matched away? Reads reports/deployment_roi/totalseg_failure_features.csv (tools/totalseg_failure_analysis.py).

Each failed scan gets up to 3 successful controls with the same scanner and label (seeded, without
replacement), so neither can explain the result. Logistic regression (standardised) and gradient-boosted
trees, 5-fold patient-grouped CV: held-out AUROC tells whether the features explain failure at all;
permutation importance on the held-out folds (drop in AUROC) ranks them. sklearn only, no SHAP dependency.

    python tools/totalseg_failure_model.py
Writes reports/deployment_roi/totalseg_failure_model.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC = "reports/deployment_roi/totalseg_failure_features.csv"
OUT = "reports/deployment_roi/totalseg_failure_model.csv"
FEATS = ["sx", "sz", "nx", "nz", "fov_mm", "length_mm", "panc_ml", "panc_to_top_mm", "panc_to_bottom_mm",
         "panc_hu_median", "hu_min", "hu_median", "hu_max", "air_frac", "age", "male", "manual_mask",
         "orient_common"]
K = 3


def main():
    d = pd.read_csv(SRC)
    d["failed"] = d["empty"].astype(str).str.lower() == "true"
    d["male"] = (d.sex == "M").astype(int)
    d["manual_mask"] = (d.mask_source == "manual").astype(int)
    d["orient_common"] = (d.orient == d.orient.mode()[0]).astype(int)
    rng = np.random.default_rng(0)
    keep = []
    for (sc, y), g in d.groupby(["scanner", "y"]):
        f, ok = g[g.failed], g[~g.failed]
        keep += list(f.index) + list(rng.choice(ok.index, min(len(ok), K * len(f)), replace=False))
    m = d.loc[keep].reset_index(drop=True)
    X, y = m[FEATS].astype(float), m.failed.astype(int).values
    groups = m.case.str.split("_").str[0]
    print(f"matched set: {y.sum()} failed + {len(y) - y.sum()} controls (same scanner and label)")
    models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "boosted_trees": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                              random_state=0)}
    rows = []
    for name, mdl in models.items():
        oof, imps = np.zeros(len(y)), []
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
            Xtr = X.iloc[tr].fillna(X.iloc[tr].median())
            Xte = X.iloc[te].fillna(X.iloc[tr].median())
            mdl.fit(Xtr, y[tr])
            oof[te] = mdl.predict_proba(Xte)[:, 1]
            imps.append(permutation_importance(mdl, Xte, y[te], scoring="roc_auc", n_repeats=20,
                                               random_state=0).importances_mean)
        auc = roc_auc_score(y, oof)
        imp = np.mean(imps, axis=0)
        print(f"\n{name}: held-out AUROC {auc:.3f}")
        for i in np.argsort(-imp)[:8]:
            print(f"  {FEATS[i]:18s} AUROC drop {imp[i]:+.3f}")
        rows += [{"model": name, "cv_auroc": auc, "feature": f, "perm_importance": v} for f, v in zip(FEATS, imp)]
        if name == "logistic":
            mdl.fit(X.fillna(X.median()), y)
            coef = mdl[-1].coef_[0]
            for r, c in zip(rows[-len(FEATS):], coef):
                r["std_coef"] = c
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
