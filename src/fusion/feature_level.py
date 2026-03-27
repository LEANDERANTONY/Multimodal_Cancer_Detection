from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split


def get_ct_embeddings(model: torch.nn.Module, loader: Any, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    model.eval()
    embeddings = []
    labels = []
    activation: dict[str, torch.Tensor] = {}

    def hook(module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> None:
        activation["out"] = output.detach()

    handle = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            _ = model(images)
            emb = activation["out"].squeeze(-1).squeeze(-1)
            embeddings.append(emb.cpu())
            labels.extend(batch_labels.numpy())
    handle.remove()
    return torch.cat(embeddings), np.asarray(labels)


def make_label_matched_fused_dataset(
    ct_emb: torch.Tensor,
    ct_lbl: np.ndarray,
    bio_emb: torch.Tensor,
    bio_lbl: np.ndarray,
    mismatch: bool = False,
    max_per_class: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(42)
    ct_np = ct_emb.numpy() if hasattr(ct_emb, "numpy") else np.asarray(ct_emb)
    bio_np = bio_emb.numpy() if hasattr(bio_emb, "numpy") else np.asarray(bio_emb)
    X_list = []
    y_list = []
    for label in [0, 1]:
        ct_idx = np.where(ct_lbl == label)[0]
        bio_target = (1 - label) if mismatch else label
        bio_idx = np.where(bio_lbl == bio_target)[0]
        n = min(len(ct_idx), len(bio_idx))
        if max_per_class is not None:
            n = min(n, max_per_class)
        if n <= 0:
            continue
        ct_sel = rng.choice(ct_idx, n, replace=False)
        bio_sel = rng.choice(bio_idx, n, replace=False)
        for ci, bi in zip(ct_sel, bio_sel):
            X_list.append(np.concatenate([ct_np[ci], bio_np[bi]]))
            y_list.append(label)
    return np.stack(X_list), np.asarray(y_list)


def evaluate_feature_level_models(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
    rows = []
    for name, clf in [
        ("LogReg", LogisticRegression(max_iter=1000)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=random_state)),
    ]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "auc": float(roc_auc_score(y_test, y_prob)),
                "acc": float(accuracy_score(y_test, y_pred)),
                "f1": float(precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)[2]),
                "n_test": int(len(y_test)),
            }
        )
    return pd.DataFrame(rows)
