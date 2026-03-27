from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from tqdm.auto import tqdm

from src.models.ct import IMAGENET_MEAN, IMAGENET_STD


def build_embedding_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_embedding(image_path: str | Path, model: torch.nn.Module, device: torch.device, transform: Any | None = None) -> np.ndarray | None:
    transform = transform or build_embedding_transform()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    tensor = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze().cpu().numpy()


def extract_ct_embeddings_from_dir(
    root_dir: str | Path,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame]:
    root_dir = Path(root_dir)
    embeddings: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []

    for label in ["cancer", "control"]:
        label_dir = root_dir / label
        if not label_dir.exists():
            continue
        patients = sorted(path for path in label_dir.iterdir() if path.is_dir())
        for patient_dir in tqdm(patients, desc=f"Embedding {label}"):
            for image_path in sorted(path for path in patient_dir.iterdir() if path.is_file()):
                embedding = get_embedding(image_path, model=model, device=device)
                if embedding is None:
                    continue
                embeddings.append(embedding)
                metadata.append({"patient": patient_dir.name, "label": label, "file": image_path.name})

    return np.array(embeddings), pd.DataFrame(metadata)


def fit_pca_embeddings(embeddings: np.ndarray, n_components: float = 0.95, random_state: int = 42) -> dict[str, Any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(scaled)
    return {"scaled": scaled, "pca_embeddings": transformed, "scaler": scaler, "pca": pca}


def compute_k_selection_metrics(
    pca_embeddings: np.ndarray,
    ks: range = range(2, 16),
    random_state: int = 42,
) -> pd.DataFrame:
    inertias = []
    silhouettes = []
    sample_size = min(5000, len(pca_embeddings))

    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(pca_embeddings)
        inertias.append(km.inertia_)
        silhouettes.append(
            silhouette_score(
                pca_embeddings,
                labels,
                sample_size=sample_size if sample_size >= 2 else None,
                random_state=random_state,
            )
        )

    return pd.DataFrame({"k": list(ks), "inertia": inertias, "silhouette": silhouettes})


def run_kmeans_assignments(
    pca_embeddings: np.ndarray,
    metadata: pd.DataFrame,
    k: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(pca_embeddings)
    output = metadata.copy()
    cluster_col = f"cluster_k{k}_cropped"
    output[cluster_col] = labels
    summary = output.groupby([cluster_col, "label"]).size().unstack(fill_value=0)
    return output, summary
