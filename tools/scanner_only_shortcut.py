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

Predictor = P(PDAC | scanner) estimated empirically, scored out-of-fold
(per-scanner rate fit on train folds only, applied to held-out cases, pooled)
with a bootstrap 95% CI. Three framings are reported, because two matter:
  - all categories: folds in the `Unknown` (missing/"0") scanner group. Non-random
    missingness (36% PDAC) is predictive but an image model CANNOT read a missing
    metadata field from pixels -- so this is "scanner field incl. missingness",
    not a pure acquisition signature.
  - known manufacturers only: drops `Unknown` -- the honest "manufacturer alone".
  - patient-grouped: PANORAMA has 2238 studies / 2224 patients (14 repeat exams);
    grouping by PANORAMA_patient_id keeps a patient out of both train and test, so
    the metric is genuinely patient-level.

No image data is read and the source dataset is not modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "data" / "raw" / "ct" / "panorama_labels" / "clinical_information.xlsx"
PATIENT_COL = "PANORAMA_patient_id"
SEED = 20261001
N_BOOT = 2000


def clean_scanner(col: pd.Series) -> pd.Series:
    """NaN, the literal string "0", and blanks collapse into an explicit Unknown."""
    s = col.astype("string")
    s = s.where(~s.isna(), "Unknown")
    s = s.where(s.str.strip() != "0", "Unknown")
    s = s.where(s.str.strip() != "", "Unknown")
    return s


def load() -> pd.DataFrame:
    try:
        df = pd.read_excel(XLSX)
    except ImportError as e:  # openpyxl is the .xlsx engine; it is a declared dependency
        raise SystemExit(
            f"Reading {XLSX.name} needs the 'openpyxl' engine ({e}). "
            "It is in pyproject.toml -- run `uv sync` (or `pip install openpyxl`)."
        )
    df["scanner_clean"] = clean_scanner(df["scanner"])
    df["y"] = (df["label"].astype(str).str.upper() == "PDAC").astype(int)
    return df


def oof_predictions(df: pd.DataFrame, group: bool) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold P(PDAC|scanner): rate fit on train only; unseen scanner -> train base rate."""
    y = df["y"].to_numpy()
    sc = df["scanner_clean"].to_numpy()
    n = len(y)
    oof = np.full(n, np.nan)
    if group:
        splits = StratifiedGroupKFold(n_splits=5).split(np.zeros(n), y, df[PATIENT_COL].to_numpy())
    else:
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(n), y)
    for tr, te in splits:
        base = y[tr].mean()
        rate = pd.Series(y[tr], index=sc[tr]).groupby(level=0).mean()
        oof[te] = [rate.get(s, base) for s in sc[te]]
    return y, oof


def report(label: str, y: np.ndarray, score: np.ndarray, rng: np.random.Generator) -> None:
    idx = np.arange(len(y))
    boot = [
        roc_auc_score(y[b], score[b])
        for b in (rng.choice(idx, len(y), replace=True) for _ in range(N_BOOT))
        if 0 < y[b].sum() < len(b)
    ]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(
        f"  {label:42s} AUROC={roc_auc_score(y, score):.4f}  "
        f"95% CI [{lo:.3f}, {hi:.3f}]  AP={average_precision_score(y, score):.4f}  "
        f"(n={len(y)} pos={int(y.sum())})"
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = load()
    n = len(df)
    print(f"n_studies={n}  patients={df[PATIENT_COL].nunique()}  "
          f"PDAC={int(df['y'].sum())} ({df['y'].mean() * 100:.1f}%)  chance AP={df['y'].mean():.4f}")

    tab = pd.crosstab(df["scanner_clean"], df["y"]).rename(columns={0: "nonPDAC", 1: "PDAC"})
    tab["total"] = tab.sum(axis=1)
    tab["pct_PDAC"] = (tab["PDAC"] / tab["total"] * 100).round(1)
    print("\nscanner x label:")
    print(tab.sort_values("pct_PDAC").to_string())

    chi2, p, dof, _ = chi2_contingency(pd.crosstab(df["scanner_clean"], df["y"]))
    print(f"\nchi2={chi2:.1f}  dof={dof}  p={p:.3e}  Cramer's V={np.sqrt(chi2 / n):.3f}")

    known = df[df["scanner_clean"] != "Unknown"]
    print("\nout-of-fold scanner-only detection ceiling:")
    y, oof = oof_predictions(df, group=False);  report("all categories, study-level (ref)", y, oof, rng)
    y, oof = oof_predictions(df, group=True);   report("all categories, patient-grouped", y, oof, rng)
    y, oof = oof_predictions(known, group=True); report("manufacturer alone (no Unknown), patient-grp", y, oof, rng)
    print("\nNote: 'all categories' folds in non-random missingness (Unknown, 36% PDAC), which an image\n"
          "model cannot read; 'manufacturer alone' is the pure acquisition-signature figure.")


if __name__ == "__main__":
    main()
