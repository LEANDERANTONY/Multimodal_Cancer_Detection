
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os

RAW_PATH = "data/raw/biomarkers/urinary_biomarkers.csv"
CLEAN_PATH = "data/processed/biomarkers_clean.csv"
SUMMARY_PATH = "data/processed/biomarkers_summary.csv"

def main():
    print("Loading biomarker dataset...")
    df = pd.read_csv(RAW_PATH)
    print(f"Initial shape: {df.shape}")

    # Report missingness
    missing_pct = df.isna().mean() * 100
    print("Missing value percentage per column:")
    print(missing_pct)

    # --- Fix Stage (Roman numerals -> int) ---
    roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    if "stage" in df.columns:
        df["stage"] = df["stage"].map(roman_map)

    # Metadata vs numeric separation
    metadata_cols = ["sample_id", "patient_cohort", "sample_origin", "sex", "diagnosis", "benign_sample_diagnosis"]
    numeric_cols = [c for c in df.columns if c not in metadata_cols]

    df_meta = df[metadata_cols]
    df_numeric = df[numeric_cols]

    # Choose imputation strategy
    if missing_pct.max() >= 10:
        print("→ Missingness >=10%. Using KNN imputation (k=5).")
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)
    else:
        print("→ Missingness <10%. Using mean imputation.")
        df_imputed = df_numeric.fillna(df_numeric.mean())

    # Normalize numeric features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=numeric_cols)
    print("→ Applied z-score normalization.")

    # Combine back
    df_clean = pd.concat([df_meta.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

    # Save cleaned dataset
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"Clean dataset saved → {CLEAN_PATH}")

    # Save summary grouped by diagnosis
    summary = df_clean.groupby("diagnosis").describe().transpose()
    summary.to_csv(SUMMARY_PATH)
    print(f"Summary saved → {SUMMARY_PATH}")

if __name__ == "__main__":
    main()


