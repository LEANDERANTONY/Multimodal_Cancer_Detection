"""Biomarker arm: the same confound audit the CT models got, on the urinary biomarker cohort (Debernardi et al.).

Task: PDAC (diagnosis 3) vs non-PDAC (1 healthy + 2 benign), as in PANORAMA; secondary: PDAC vs healthy.
Models (repeated stratified 5-fold CV, 10 repeats, preprocessing fitted inside each fold; L2 logistic regression):
    site-only  one-hot sample_origin         (how much the collection site alone predicts the label)
    age-only   age
    panel      log LYVE1, REG1B, TFF1, creatinine (the published panel, without age)
    panel+age  the published model
Per model: pooled out-of-fold AUROC (mean over repeats) and, for the panel models, within-site AUROC (sites with
>= 8 cases of each class) and the site tax = mixed AUROC - mean within-site AUROC, with a bootstrap CI
(stratified by site x label). Leave-one-site-out: train on the other sites, test on the held-out one.

    python tools/biomarker_confound.py
Writes reports/biomarker_confound.csv.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "data/raw/biomarkers/urinary_biomarkers.csv"
OUT = "reports/biomarker_confound.csv"
PANEL = ["LYVE1", "REG1B", "TFF1", "creatinine"]
N_BOOT = 1000


def features(d, kind):
    if kind == "site-only":
        return OneHotEncoder(sparse_output=False).fit_transform(d[["sample_origin"]])
    cols = {"age-only": [], "panel": PANEL, "panel+age": PANEL}[kind]
    x = np.log1p(d[cols].values) if cols else np.empty((len(d), 0))
    if kind in ("age-only", "panel+age"):
        x = np.column_stack([x, d.age.values])
    return x


def oof(x, y, strat, seed=0):
    """Out-of-fold scores, averaged over 10 repeats of stratified 5-fold CV."""
    p, n = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed).split(x, strat):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x[tr], y[tr])
        p[te] += m.predict_proba(x[te])[:, 1]
        n[te] += 1
    return p / n


def site_tax(d, p):
    sites = [s for s, g in d.groupby("sample_origin") if g.y.sum() >= 8 and (1 - g.y).sum() >= 8]
    within = {s: roc_auc_score(d.y[d.sample_origin == s], p[(d.sample_origin == s).values]) for s in sites}
    mixed = roc_auc_score(d.y, p)
    return mixed - np.mean(list(within.values())), mixed, within


def main():
    raw = pd.read_csv(DATA)
    rows = []
    for task, d in (("pdac_vs_rest", raw.assign(y=(raw.diagnosis == 3).astype(int))),
                    ("pdac_vs_healthy", raw[raw.diagnosis != 2].assign(y=lambda e: (e.diagnosis == 3).astype(int)))):
        d = d.reset_index(drop=True)
        strat = d.sample_origin + d.y.astype(str)
        for kind in ("site-only", "age-only", "panel", "panel+age"):
            x = features(d, kind)
            p = oof(x, d.y.values, d.y.values if kind == "site-only" else strat)
            r = {"task": task, "model": kind, "n": len(d), "pos": int(d.y.sum()), "auroc": roc_auc_score(d.y, p)}
            if kind.startswith("panel"):
                t, mixed, within = site_tax(d, p)
                r.update(tax=t, **{f"within_{s}": v for s, v in within.items()})
                rng, cells = np.random.default_rng(0), d.groupby(strat.values).indices
                bs = []
                for _ in range(N_BOOT):
                    idx = np.concatenate([rng.choice(ix, len(ix)) for ix in cells.values()])
                    bs.append(site_tax(d.iloc[idx].reset_index(drop=True), p[idx])[0])
                r.update(tax_lo=np.percentile(bs, 2.5), tax_hi=np.percentile(bs, 97.5))
                # leave-one-site-out
                for s in d.sample_origin.unique():
                    te = (d.sample_origin == s).values
                    if d.y[te].nunique() < 2 or d.y[~te].nunique() < 2:
                        continue
                    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x[~te], d.y[~te])
                    r[f"loso_{s}"] = roc_auc_score(d.y[te], m.predict_proba(x[te])[:, 1])
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
