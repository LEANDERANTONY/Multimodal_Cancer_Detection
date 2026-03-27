from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def build_cluster_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    val_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    df_pat = df.groupby("patient").agg(label=("label", "first")).reset_index()
    y = df_pat["label"].map({"control": 0, "cancer": 1})

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(df_pat, y))
    df_temp_pat = df_pat.iloc[temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / test_size, random_state=random_state)
    y_temp = df_temp_pat["label"].map({"control": 0, "cancer": 1})
    val_idx, test_idx = next(sss2.split(df_temp_pat, y_temp))

    df_val_pat = df_temp_pat.iloc[val_idx]
    df_test_pat = df_temp_pat.iloc[test_idx]

    df_out = df.copy()
    df_out["split"] = "train"
    df_out.loc[df_out["patient"].isin(df_val_pat["patient"]), "split"] = "val"
    df_out.loc[df_out["patient"].isin(df_test_pat["patient"]), "split"] = "test"
    return df_out


def split_clustered_dataframe(
    df_all: pd.DataFrame,
    cluster_column: str = "cluster_k2_cropped",
    clusters: tuple[int, ...] = (0, 1),
    random_state: int = 42,
) -> dict[int, pd.DataFrame]:
    outputs: dict[int, pd.DataFrame] = {}
    for cluster in clusters:
        df_cluster = df_all[df_all[cluster_column] == cluster].reset_index(drop=True)
        outputs[cluster] = build_cluster_split(df_cluster, random_state=random_state)
    return outputs


def summarize_cluster_predictions(results: dict[str, Any]) -> pd.DataFrame:
    df_test = pd.DataFrame(
        {
            "label": results["labels"],
            "pred": results["preds"],
            "prob": results["probs"],
            "cluster": results["meta"]["cluster_k2"],
        }
    )
    df_test["cluster"] = df_test["cluster"].astype(int)

    rows = []
    for cluster in sorted(df_test["cluster"].unique()):
        sub = df_test[df_test["cluster"] == cluster]
        acc = accuracy_score(sub["label"], sub["pred"])
        prec, rec, f1, _ = precision_recall_fscore_support(
            sub["label"], sub["pred"], average="binary", zero_division=0
        )
        try:
            auc_val = roc_auc_score(sub["label"], sub["prob"])
        except Exception:
            auc_val = 0.5
        rows.append(
            {
                "cluster": cluster,
                "n_slices": len(sub),
                "acc": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc_val,
            }
        )
    return pd.DataFrame(rows)


def make_ct_results_summary(
    model_name: str,
    best_epoch: int,
    test_results: dict[str, Any],
    test_patient: dict[str, Any],
    cluster_results: list[dict[str, Any]] | None = None,
    data_version: str = "ct_cropped",
    preprocessing: str = "standardized + CLAHE (train only)",
    clustering: str = "kmeans_k2",
) -> dict[str, Any]:
    return {
        "model": model_name,
        "data_version": data_version,
        "preprocessing": preprocessing,
        "clustering": clustering,
        "best_epoch": int(best_epoch),
        "slice_level": {
            "accuracy": float(test_results["acc"]),
            "precision": float(test_results["precision"]),
            "recall": float(test_results["recall"]),
            "f1": float(test_results["f1"]),
            "auc": float(test_results["auc"]),
        },
        "patient_level": {
            "n_patients": int(test_patient["n_patients"]),
            "accuracy": float(test_patient["acc"]),
            "precision": float(test_patient["precision"]),
            "recall": float(test_patient["recall"]),
            "f1": float(test_patient["f1"]),
            "auc": float(test_patient["auc"]),
        },
        "cluster_k2": cluster_results,
    }


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
