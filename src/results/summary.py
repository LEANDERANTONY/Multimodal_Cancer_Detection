from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def load_json_if_exists(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_ct_metrics(
    reports_dir: str | Path,
    in_memory_test_results: dict[str, Any] | None = None,
    in_memory_test_patient: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    reports_dir = Path(reports_dir)

    if in_memory_test_results is not None and in_memory_test_patient is not None:
        return {
            "slice_level": {
                "auc": float(in_memory_test_results.get("auc", float("nan"))),
                "accuracy": float(in_memory_test_results.get("acc", float("nan"))),
                "f1": float(in_memory_test_results.get("f1", float("nan"))),
            },
            "patient_level": {
                "auc": float(in_memory_test_patient.get("auc", float("nan"))),
                "accuracy": float(in_memory_test_patient.get("acc", float("nan"))),
                "f1": float(in_memory_test_patient.get("f1", float("nan"))),
            },
            "source": "in-memory test_results/test_patient",
            "architecture": "ResNet50 (global)",
        }

    comparison_path = reports_dir / "final_model_comparison.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        if "Model" in df.columns and len(df) > 0:
            row = df.loc[df["Model"].str.lower() == "global"]
            row = df.iloc[[0]] if row.empty else row.iloc[[0]]
            record = row.iloc[0]
            return {
                "slice_level": {
                    "auc": float(record.get("Slice_AUC", float("nan"))),
                    "accuracy": float(record.get("Slice_Acc", float("nan"))),
                    "f1": float(record.get("Slice_F1", float("nan"))),
                },
                "patient_level": {
                    "auc": float(record.get("Patient_AUC", float("nan"))),
                    "accuracy": float(record.get("Patient_Acc", float("nan"))),
                    "f1": float(record.get("Patient_F1", float("nan"))),
                },
                "source": comparison_path.name,
                "architecture": "ResNet50 (global)",
            }

    legacy = load_json_if_exists(reports_dir / "ct_model_results.json")
    if legacy:
        return {
            "slice_level": {
                "auc": float(legacy.get("slice_level", {}).get("auc", float("nan"))),
                "accuracy": float(legacy.get("slice_level", {}).get("accuracy", float("nan"))),
                "f1": float(legacy.get("slice_level", {}).get("f1", float("nan"))),
            },
            "patient_level": {
                "auc": float(legacy.get("patient_level", {}).get("auc", float("nan"))),
                "accuracy": float(legacy.get("patient_level", {}).get("accuracy", float("nan"))),
                "f1": float(legacy.get("patient_level", {}).get("f1", float("nan"))),
            },
            "source": "ct_model_results.json",
            "architecture": legacy.get("model", "CT classifier"),
        }

    return None


def load_biomarker_metrics(
    reports_dir: str | Path,
    in_memory_metrics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    reports_dir = Path(reports_dir)

    if in_memory_metrics is not None:
        return {
            "auc": float(in_memory_metrics.get("auc", float("nan"))),
            "accuracy": float(in_memory_metrics.get("accuracy", float("nan"))),
            "f1": float(in_memory_metrics.get("f1", float("nan"))),
            "source": "in-memory biomarker metrics",
        }

    legacy = load_json_if_exists(reports_dir / "biomarker_results.json")
    if legacy:
        return {
            "auc": float(legacy.get("auc", float("nan"))),
            "accuracy": float(legacy.get("accuracy", float("nan"))),
            "f1": float(legacy.get("f1", float("nan"))),
            "source": "biomarker_results.json",
        }

    return None


def build_final_summary(
    ct_metrics: dict[str, Any] | None,
    skip_biomarker: bool = False,
    bio_metrics: dict[str, Any] | None = None,
    biomarker_features: Sequence[str] | str | None = None,
    experiment_date: str | None = None,
) -> dict[str, Any]:
    summary = {
        "experiment_date": experiment_date or pd.Timestamp.now().isoformat(),
        "ct_model": {
            "architecture": (ct_metrics or {}).get("architecture", "CT classifier"),
            "preprocessing": "Bias-mitigation + orientation/cropping + ImageNet normalization (see Phase 1/2)",
            "test_metrics": {
                "slice_level": (ct_metrics or {}).get("slice_level", {}),
                "patient_level": (ct_metrics or {}).get("patient_level", {}),
            },
            "source": (ct_metrics or {}).get("source", "unknown"),
        },
    }

    if not skip_biomarker:
        summary["biomarker_model"] = {
            "architecture": "MLP (64-32)",
            "features": biomarker_features if biomarker_features is not None else "unknown",
            "test_metrics": {
                "auc": (bio_metrics or {}).get("auc", float("nan")),
                "accuracy": (bio_metrics or {}).get("accuracy", float("nan")),
                "f1": (bio_metrics or {}).get("f1", float("nan")),
            },
            "source": (bio_metrics or {}).get("source", "unknown"),
        }
        summary["fusion_note"] = (
            "Synthetic pairing used (different patient cohorts). "
            "CT retains residual domain signal; fusion results are exploratory and should not be interpreted as clinically validated performance."
        )

    return summary


def build_model_comparison_table(
    ct_metrics: dict[str, Any] | None,
    skip_biomarker: bool = False,
    bio_metrics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows = []

    def safe_fmt(value: Any) -> str:
        try:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return "NA"
            return f"{float(value):.3f}"
        except Exception:
            return "NA"

    if ct_metrics:
        rows.append(
            {
                "Model": "CT (Global)",
                "Modality": "CT Images",
                "AUC": safe_fmt(ct_metrics["patient_level"].get("auc")),
                "Accuracy": safe_fmt(ct_metrics["patient_level"].get("accuracy")),
                "F1": safe_fmt(ct_metrics["patient_level"].get("f1")),
                "Note": "Patient-level aggregation",
            }
        )
    else:
        rows.append(
            {
                "Model": "CT (Global)",
                "Modality": "CT Images",
                "AUC": "NA",
                "Accuracy": "NA",
                "F1": "NA",
                "Note": "Metrics not found in saved reports",
            }
        )

    if not skip_biomarker:
        rows.append(
            {
                "Model": "Biomarker (MLP)",
                "Modality": "Urinary Biomarkers",
                "AUC": safe_fmt((bio_metrics or {}).get("auc")),
                "Accuracy": safe_fmt((bio_metrics or {}).get("accuracy")),
                "F1": safe_fmt((bio_metrics or {}).get("f1")),
                "Note": "7 biomarker features",
            }
        )

    return pd.DataFrame(rows)
