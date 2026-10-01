from __future__ import annotations

"""Scanner-only shortcut baseline for the PANORAMA PDAC label.

Measures how well the CT *scanner manufacturer alone* -- no pixels, no `level`,
no demographics -- predicts the patient-level PDAC label. This is the honest
"acquisition-signature shortcut ceiling": the slice of detection performance an
image model can obtain for free by reading scanner fingerprints, rather than
tumour biology.

It complements ``reports/panorama_confound_audit.md``, which reported scanner's
association (Cramer's V = 0.44) and a full-metadata AUC of 0.90 that was
dominated by the near-label-leakage ``level`` column. Scanner alone is the
physically-meaningful, deployable-shortcut number.

Predictor = P(PDAC | scanner) estimated empirically. Reports:
  (1) in-sample AUROC/AP  -- the association as measured on this cohort;
  (2) out-of-fold AUROC/AP -- per-scanner rate fit on train folds only, applied
      to held-out cases and pooled, with a bootstrap 95% CI. This is the number
      to report; it carries no in-sample optimism.

No image data is read and the source dataset is not modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "data" / "raw" / "ct" / "panorama_labels" / "clinical_information.xlsx"
SEED = 20261001
N_BOOT = 2000


def clean_scanner(col: pd.Series) -> pd.Series:
    """NaN, the literal string "0", and blanks collapse into an explicit Unknown."""
    s = col.astype("string")
    s = s.where(~s.isna(), "Unknown")
    s = s.where(s.str.strip() != "0", "Unknown")
    s = s.where(s.str.strip() != "", "Unknown")
    return s


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_excel(XLSX)
    df["scanner_clean"] = clean_scanner(df["scanner"])
    y = (df["label"].astype(str).str.upper() == "PDAC").astype(int).to_numpy()
    scanners = df["scanner_clean"].to_numpy()
    n = len(y)
    print(f"n={n}  PDAC={int(y.sum())} ({y.mean() * 100:.1f}%)  non-PDAC={n - int(y.sum())}")

    tab = pd.crosstab(df["scanner_clean"], y).rename(columns={0: "nonPDAC", 1: "PDAC"})
    tab["total"] = tab.sum(axis=1)
    tab["pct_PDAC"] = (tab["PDAC"] / tab["total"] * 100).round(1)
    print("\nscanner x label:")
    print(tab.sort_values("pct_PDAC").to_string())

    chi2, p, dof, _ = chi2_contingency(pd.crosstab(df["scanner_clean"], y))
    cramers_v = float(np.sqrt(chi2 / n))  # k=2 label columns -> min(r-1,c-1)=1
    print(f"\nchi2={chi2:.1f}  dof={dof}  p={p:.3e}  Cramer's V={cramers_v:.3f}")

    # (1) in-sample: each case scored by its scanner's empirical PDAC rate.
    rate = pd.Series(y, index=scanners).groupby(level=0).mean()
    score_in = np.array([rate[s] for s in scanners])
    print(
        f"\n[in-sample]   scanner-only  AUROC={roc_auc_score(y, score_in):.4f}  "
        f"AP={average_precision_score(y, score_in):.4f}  (chance AP={y.mean():.4f})"
    )

    # (2) out-of-fold: rate learned on train only; unseen scanner -> global train rate.
    oof = np.full(n, np.nan)
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(n), y):
        global_rate = y[tr].mean()
        r = pd.Series(y[tr], index=scanners[tr]).groupby(level=0).mean()
        oof[te] = [r.get(s, global_rate) for s in scanners[te]]

    boot = []
    idx = np.arange(n)
    for _ in range(N_BOOT):
        b = rng.choice(idx, n, replace=True)
        if 0 < y[b].sum() < len(b):
            boot.append(roc_auc_score(y[b], oof[b]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(
        f"[out-of-fold] scanner-only  AUROC={roc_auc_score(y, oof):.4f}  "
        f"95% CI [{lo:.3f}, {hi:.3f}]  AP={average_precision_score(y, oof):.4f}"
    )


if __name__ == "__main__":
    main()
