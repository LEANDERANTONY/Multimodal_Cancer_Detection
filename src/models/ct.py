from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, precision_recall_curve, roc_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm.auto import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CTSliceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        root_dir: str | Path,
        transforms: Any | None = None,
        cluster_column: str = "cluster_k2_cropped",
    ) -> None:
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.cluster_column = cluster_column
        self.label_map = {"control": 0, "cancer": 1}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.df.iloc[index]
        label = str(row["label"])
        patient = str(row["patient"])
        file_name = str(row["file"])
        image_path = self.root_dir / label / patient / file_name

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        cluster_value = row[self.cluster_column] if self.cluster_column in row.index else -1
        meta = {
            "patient": patient,
            "file": file_name,
            "path": str(image_path),
            "cluster_k2": int(cluster_value) if pd.notna(cluster_value) else -1,
        }
        return image, torch.tensor(self.label_map[label], dtype=torch.long), meta


def build_ct_transforms(train: bool = True) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    ],
                    p=0.3,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    class_counts = np.bincount(labels.astype(int))
    return torch.tensor([len(labels) / (2 * count) for count in class_counts], dtype=torch.float32)


def create_ct_dataloaders(
    df_split: pd.DataFrame,
    root_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 2,
) -> dict[str, Any]:
    train_ds = CTSliceDataset(df_split, "train", root_dir=root_dir, transforms=build_ct_transforms(train=True))
    val_ds = CTSliceDataset(df_split, "val", root_dir=root_dir, transforms=build_ct_transforms(train=False))
    test_ds = CTSliceDataset(df_split, "test", root_dir=root_dir, transforms=build_ct_transforms(train=False))

    train_labels = train_ds.df["label"].map({"control": 0, "cancer": 1}).values
    class_weights = compute_class_weights(train_labels)

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        ),
        "datasets": {"train": train_ds, "val": val_ds, "test": test_ds},
        "class_weights": class_weights,
    }
    return dataloaders


def build_resnet50_classifier(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
    return model


def build_resnet50_feature_extractor(pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    return nn.Sequential(*list(model.children())[:-1])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []

    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())

    epoch_loss = running_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc_val = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc_val = 0.5
    return epoch_loss, acc, auc_val


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []
    all_meta = {"patient": [], "file": [], "path": [], "cluster_k2": []}

    with torch.no_grad():
        for images, labels, meta in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_meta["patient"].extend(meta["patient"])
            all_meta["file"].extend(meta["file"])
            all_meta["path"].extend(meta["path"])
            all_meta["cluster_k2"].extend(int(x) for x in meta["cluster_k2"])

    epoch_loss = running_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    try:
        auc_val = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc_val = 0.5

    return {
        "loss": epoch_loss,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_val,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "meta": all_meta,
    }


def patient_level_metrics(results: dict[str, Any]) -> dict[str, Any]:
    df = pd.DataFrame(
        {
            "patient": results["meta"]["patient"],
            "label": results["labels"],
            "prob": results["probs"],
        }
    )
    patient_df = df.groupby("patient").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()
    patient_df["pred"] = (patient_df["prob"] >= 0.5).astype(int)

    acc = accuracy_score(patient_df["label"], patient_df["pred"])
    prec, rec, f1, _ = precision_recall_fscore_support(
        patient_df["label"], patient_df["pred"], average="binary", zero_division=0
    )
    try:
        auc_val = roc_auc_score(patient_df["label"], patient_df["prob"])
    except Exception:
        auc_val = 0.5

    return {
        "n_patients": len(patient_df),
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_val,
        "df": patient_df,
    }


@dataclass(frozen=True)
class CTCheckpoint:
    epoch: int | None
    val_auc: float | None
    val_acc: float | None
    path: Path


@dataclass(frozen=True)
class CTTrainingConfig:
    epochs: int = 20
    lr: float = 1e-4
    patience: int = 6
    weight_decay: float = 1e-4
    checkpoint_name: str = "ct_resnet50.pt"


@dataclass
class CTTrainingRun:
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    history: dict[str, list[float]]
    best_epoch: int
    best_val_auc: float
    checkpoint_path: Path


def load_ct_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> CTCheckpoint:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return CTCheckpoint(
        epoch=checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        val_auc=checkpoint.get("val_auc") if isinstance(checkpoint, dict) else None,
        val_acc=checkpoint.get("val_acc") if isinstance(checkpoint, dict) else None,
        path=checkpoint_path,
    )


def train_ct_model(
    model: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    models_dir: str | Path,
    config: CTTrainingConfig | None = None,
) -> CTTrainingRun:
    config = config or CTTrainingConfig()
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    checkpoint_path = models_dir / config.checkpoint_name

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        train_loss, train_acc, train_auc = train_epoch(model, dl_train, criterion, optimizer, device)
        val_results = evaluate(model, dl_val, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_results["loss"])
        history["val_acc"].append(val_results["acc"])
        history["val_auc"].append(val_results["auc"])

        if val_results["auc"] > best_val_auc:
            best_val_auc = float(val_results["auc"])
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": float(val_results["auc"]),
                    "val_acc": float(val_results["acc"]),
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    return CTTrainingRun(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        best_epoch=best_epoch,
        best_val_auc=best_val_auc,
        checkpoint_path=checkpoint_path,
    )


def plot_training_history(history: dict[str, list[float]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="Train", marker="o")
    axes[0].plot(history["val_loss"], label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", marker="o")
    axes[1].plot(history["val_acc"], label="Val", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["train_auc"], label="Train", marker="o")
    axes[2].plot(history["val_auc"], label="Val", marker="s")
    axes[2].axhline(y=0.5, linestyle="--", label="Random")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].set_title("AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_test_curves(results: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fpr, tpr, _ = roc_curve(results["labels"], results["probs"])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "r--")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(results["labels"], results["probs"])
    pr_auc = auc(rec, prec)
    axes[1].plot(rec, prec, label=f"AP = {pr_auc:.3f}")
    axes[1].set_title("Precision-Recall")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    cm = confusion_matrix(results["labels"], results["preds"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Control", "Cancer"])
    disp.plot(ax=axes[2], cmap="Blues", values_format="d")
    axes[2].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
