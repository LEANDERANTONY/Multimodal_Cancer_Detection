from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def collect_ct_probabilities(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all: list[float] = []
    labels_all: list[int] = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]
            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(labels.numpy())
    return np.asarray(probs_all), np.asarray(labels_all)


def build_synthetic_decision_pairs(
    ct_probs: np.ndarray,
    ct_labels: np.ndarray,
    bio_probs: np.ndarray,
    bio_labels: np.ndarray,
    mismatch: bool = False,
    max_per_class: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(42)
    fusion_probs_ct = []
    fusion_probs_bio = []
    fusion_labels = []

    for label in [0, 1]:
        ct_idx = np.where(ct_labels == label)[0]
        bio_target = (1 - label) if mismatch else label
        bio_idx = np.where(bio_labels == bio_target)[0]
        n_match = min(len(ct_idx), len(bio_idx))
        if max_per_class is not None:
            n_match = min(n_match, max_per_class)
        if n_match <= 0:
            continue

        ct_sel = rng.choice(ct_idx, n_match, replace=False)
        bio_sel = rng.choice(bio_idx, n_match, replace=False)
        fusion_probs_ct.extend(ct_probs[ct_sel])
        fusion_probs_bio.extend(bio_probs[bio_sel])
        fusion_labels.extend([label] * n_match)

    return (
        np.asarray(fusion_probs_ct),
        np.asarray(fusion_probs_bio),
        np.asarray(fusion_labels),
    )


def evaluate_weighted_decision_fusion(
    fusion_probs_ct: np.ndarray,
    fusion_probs_bio: np.ndarray,
    fusion_labels: np.ndarray,
    w_ct: float,
) -> dict[str, float | int]:
    w_bio = 1.0 - w_ct
    fused_probs = w_ct * fusion_probs_ct + w_bio * fusion_probs_bio
    fused_preds = (fused_probs >= 0.5).astype(int)
    auc_value = float("nan")
    if len(np.unique(fusion_labels)) > 1:
        auc_value = float(roc_auc_score(fusion_labels, fused_probs))

    return {
        "w_ct": float(w_ct),
        "w_bio": float(w_bio),
        "auc": auc_value,
        "acc": float(accuracy_score(fusion_labels, fused_preds)),
        "f1": float(
            precision_recall_fscore_support(
                fusion_labels,
                fused_preds,
                average="binary",
                zero_division=0,
            )[2]
        ),
        "n": int(len(fusion_labels)),
    }


def run_weighted_decision_fusion(
    ct_probs: np.ndarray,
    ct_labels: np.ndarray,
    bio_probs: np.ndarray,
    bio_labels: np.ndarray,
    weights: Sequence[float] = (0.3, 0.5, 0.7),
    mismatch: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    max_per_class = min(len(ct_probs), len(bio_probs)) // 2
    fusion_probs_ct, fusion_probs_bio, fusion_labels = build_synthetic_decision_pairs(
        ct_probs=ct_probs,
        ct_labels=ct_labels,
        bio_probs=bio_probs,
        bio_labels=bio_labels,
        mismatch=mismatch,
        max_per_class=max_per_class,
        rng=rng,
    )
    rows = [
        evaluate_weighted_decision_fusion(
            fusion_probs_ct=fusion_probs_ct,
            fusion_probs_bio=fusion_probs_bio,
            fusion_labels=fusion_labels,
            w_ct=w_ct,
        )
        for w_ct in weights
    ]
    return pd.DataFrame(rows).sort_values("w_ct"), {
        "fusion_probs_ct": fusion_probs_ct,
        "fusion_probs_bio": fusion_probs_bio,
        "fusion_labels": fusion_labels,
    }


def run_decision_fusion_sanity_check(
    ct_probs: np.ndarray,
    ct_labels: np.ndarray,
    bio_probs: np.ndarray,
    bio_labels: np.ndarray,
    seeds: Sequence[int],
    weights: Sequence[float] = (0.3, 0.5, 0.7),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    max_per_class = min(len(ct_probs), len(bio_probs)) // 2

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for mismatch in [False, True]:
            fusion_probs_ct, fusion_probs_bio, fusion_labels = build_synthetic_decision_pairs(
                ct_probs=ct_probs,
                ct_labels=ct_labels,
                bio_probs=bio_probs,
                bio_labels=bio_labels,
                mismatch=mismatch,
                max_per_class=max_per_class,
                rng=rng,
            )
            for w_ct in weights:
                row = evaluate_weighted_decision_fusion(
                    fusion_probs_ct=fusion_probs_ct,
                    fusion_probs_bio=fusion_probs_bio,
                    fusion_labels=fusion_labels,
                    w_ct=w_ct,
                )
                row["seed"] = int(seed)
                row["pairing"] = "label-matched" if not mismatch else "label-mismatch (control)"
                rows.append(row)

    by_seed = pd.DataFrame(rows)
    summary = (
        by_seed.groupby(["pairing", "w_ct"], as_index=False)
        .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"), n_mean=("n", "mean"))
        .sort_values(["pairing", "w_ct"])
    )
    return by_seed, summary
