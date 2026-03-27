import json
from pathlib import Path

from src.results.summary import (
    build_final_summary,
    build_model_comparison_table,
    load_biomarker_metrics,
    load_ct_metrics,
    load_json_if_exists,
    save_json,
)


def test_save_json_and_load_json_if_exists_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "sample.json"
    payload = {"a": 1, "b": "two"}
    save_json(payload, path)
    assert load_json_if_exists(path) == payload


def test_load_ct_metrics_prefers_in_memory_metrics(tmp_path: Path) -> None:
    metrics = load_ct_metrics(
        reports_dir=tmp_path,
        in_memory_test_results={"auc": 0.9, "acc": 0.8, "f1": 0.7},
        in_memory_test_patient={"auc": 0.95, "acc": 0.85, "f1": 0.75},
    )
    assert metrics is not None
    assert metrics["source"] == "in-memory test_results/test_patient"
    assert metrics["patient_level"]["auc"] == 0.95


def test_load_ct_metrics_falls_back_to_final_model_comparison(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "final_model_comparison.csv").write_text(
        "Model,Slice_AUC,Slice_Acc,Slice_F1,Patient_AUC,Patient_Acc,Patient_F1\n"
        "Global,0.91,0.81,0.71,0.96,0.86,0.76\n",
        encoding="utf-8",
    )
    metrics = load_ct_metrics(reports_dir)
    assert metrics is not None
    assert metrics["source"] == "final_model_comparison.csv"
    assert metrics["slice_level"]["auc"] == 0.91


def test_load_biomarker_metrics_falls_back_to_json(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "biomarker_results.json").write_text(
        json.dumps({"auc": 0.8, "accuracy": 0.7, "f1": 0.6}),
        encoding="utf-8",
    )
    metrics = load_biomarker_metrics(reports_dir)
    assert metrics is not None
    assert metrics["source"] == "biomarker_results.json"
    assert metrics["auc"] == 0.8


def test_build_final_summary_includes_biomarker_section_when_enabled() -> None:
    summary = build_final_summary(
        ct_metrics={"architecture": "ResNet50", "slice_level": {}, "patient_level": {}, "source": "demo"},
        skip_biomarker=False,
        bio_metrics={"auc": 0.8, "accuracy": 0.7, "f1": 0.6, "source": "bio"},
        biomarker_features=["feat1", "feat2"],
        experiment_date="2026-03-27T00:00:00",
    )
    assert summary["experiment_date"] == "2026-03-27T00:00:00"
    assert "biomarker_model" in summary
    assert summary["biomarker_model"]["features"] == ["feat1", "feat2"]


def test_build_model_comparison_table_formats_rows() -> None:
    df = build_model_comparison_table(
        ct_metrics={"patient_level": {"auc": 0.9, "accuracy": 0.8, "f1": 0.7}},
        skip_biomarker=False,
        bio_metrics={"auc": 0.6, "accuracy": 0.5, "f1": 0.4},
    )
    assert df.shape == (2, 6)
    assert df.iloc[0]["Model"] == "CT (Global)"
    assert df.iloc[1]["Model"] == "Biomarker (MLP)"
