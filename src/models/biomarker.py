from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class BiomarkerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


@dataclass(frozen=True)
class BiomarkerDatasetBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def create_biomarker_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> BiomarkerDatasetBundle:
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    return BiomarkerDatasetBundle(
        train=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        val=DataLoader(val_ds, batch_size=batch_size),
        test=DataLoader(test_ds, batch_size=batch_size),
    )


def infer_probs(model: nn.Module, x_np: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        xb = torch.tensor(x_np, dtype=torch.float32).to(device)
        out = model(xb)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    return probs


def evaluate_biomarker_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | np.ndarray]:
    model.eval()
    probs_all = []
    labels_all = []
    preds_all = []
    loss_total = 0.0
    n_total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            prob = F.softmax(out, dim=1)[:, 1]
            pred = out.argmax(dim=1)

            probs_all.extend(prob.cpu().numpy())
            labels_all.extend(yb.cpu().numpy())
            preds_all.extend(pred.cpu().numpy())
            loss_total += loss.item() * xb.size(0)
            n_total += xb.size(0)

    labels_np = np.asarray(labels_all)
    probs_np = np.asarray(probs_all)
    preds_np = np.asarray(preds_all)

    acc = accuracy_score(labels_np, preds_np)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_np, preds_np, average="binary", zero_division=0)
    try:
        auc_val = roc_auc_score(labels_np, probs_np)
    except Exception:
        auc_val = float("nan")

    return {
        "loss": float(loss_total / max(1, n_total)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc_val),
        "labels": labels_np,
        "preds": preds_np,
        "probs": probs_np,
    }
