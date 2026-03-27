import numpy as np

from src.fusion.decision_level import (
    build_synthetic_decision_pairs,
    evaluate_weighted_decision_fusion,
    run_decision_fusion_sanity_check,
    run_weighted_decision_fusion,
)


def test_build_synthetic_decision_pairs_respects_matching_labels() -> None:
    ct_probs = np.array([0.1, 0.2, 0.8, 0.9])
    ct_labels = np.array([0, 0, 1, 1])
    bio_probs = np.array([0.3, 0.4, 0.6, 0.7])
    bio_labels = np.array([0, 0, 1, 1])

    probs_ct, probs_bio, labels = build_synthetic_decision_pairs(
        ct_probs=ct_probs,
        ct_labels=ct_labels,
        bio_probs=bio_probs,
        bio_labels=bio_labels,
        mismatch=False,
        rng=np.random.default_rng(42),
    )

    assert probs_ct.shape == probs_bio.shape == labels.shape
    assert labels.tolist().count(0) == 2
    assert labels.tolist().count(1) == 2


def test_evaluate_weighted_decision_fusion_returns_metrics() -> None:
    metrics = evaluate_weighted_decision_fusion(
        fusion_probs_ct=np.array([0.1, 0.2, 0.8, 0.9]),
        fusion_probs_bio=np.array([0.2, 0.3, 0.7, 0.8]),
        fusion_labels=np.array([0, 0, 1, 1]),
        w_ct=0.5,
    )

    assert metrics["n"] == 4
    assert 0.0 <= metrics["acc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0


def test_run_weighted_decision_fusion_returns_sorted_dataframe() -> None:
    df, paired = run_weighted_decision_fusion(
        ct_probs=np.array([0.1, 0.2, 0.8, 0.9]),
        ct_labels=np.array([0, 0, 1, 1]),
        bio_probs=np.array([0.3, 0.4, 0.6, 0.7]),
        bio_labels=np.array([0, 0, 1, 1]),
        weights=(0.7, 0.3, 0.5),
        rng=np.random.default_rng(7),
    )

    assert df["w_ct"].tolist() == [0.3, 0.5, 0.7]
    assert paired["fusion_labels"].shape[0] == 4


def test_run_decision_fusion_sanity_check_includes_both_pairing_modes() -> None:
    by_seed, summary = run_decision_fusion_sanity_check(
        ct_probs=np.array([0.1, 0.2, 0.8, 0.9]),
        ct_labels=np.array([0, 0, 1, 1]),
        bio_probs=np.array([0.3, 0.4, 0.6, 0.7]),
        bio_labels=np.array([0, 0, 1, 1]),
        seeds=[10, 11],
        weights=(0.3, 0.7),
    )

    assert set(by_seed["pairing"]) == {"label-matched", "label-mismatch (control)"}
    assert set(summary["pairing"]) == {"label-matched", "label-mismatch (control)"}
