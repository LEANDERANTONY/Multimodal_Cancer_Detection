from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.utils.project import ProjectPaths, as_path, fmt_path


TARGET_MEAN = 128.0
TARGET_STD = 40.0


@dataclass(frozen=True)
class PixelStatsSummary:
    mean: float
    std: float
    median: float
    p5: float
    p95: float
    max: float


def resolve_image_path(path_value: str | Path, project: ProjectPaths) -> str | float:
    if pd.isna(path_value):
        return path_value

    raw = str(path_value)
    try:
        candidate = Path(raw)
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass

    normalized = raw.replace("\\", "/")
    if "ct_images" in normalized:
        suffix = normalized.split("ct_images", 1)[1].lstrip("/")
        return str(project.ct_images_dir / suffix)

    if not (":/" in normalized or normalized.startswith("/")):
        return str((project.project_root / normalized).resolve())

    return raw


def compute_pixel_summary(image: np.ndarray) -> PixelStatsSummary:
    pixels = image.astype(np.float32).reshape(-1)
    return PixelStatsSummary(
        mean=float(np.mean(pixels)),
        std=float(np.std(pixels)),
        median=float(np.median(pixels)),
        p5=float(np.percentile(pixels, 5)),
        p95=float(np.percentile(pixels, 95)),
        max=float(np.max(pixels)),
    )


def analyze_pixel_distributions(
    df: pd.DataFrame,
    n_samples: int = 500,
    random_state: int = 42,
    image_column: str = "image_path",
) -> dict[str, pd.DataFrame]:
    stats: dict[str, list[dict[str, float]]] = {"cancer": [], "control": []}

    for label in ["cancer", "control"]:
        subset_df = df[df["label"] == label]
        if subset_df.empty:
            continue

        subset = subset_df.sample(n=min(n_samples, len(subset_df)), random_state=random_state)
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Analyzing {label}"):
            img_path = row.get(image_column, None)
            if pd.isna(img_path):
                continue
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            summary = compute_pixel_summary(image)
            stats[label].append(summary.__dict__)

    return {label: pd.DataFrame(items) for label, items in stats.items()}


def compute_reference_histogram(
    df: pd.DataFrame,
    label: str = "control",
    n_samples: int = 200,
    random_state: int = 42,
    image_column: str = "image_path",
) -> np.ndarray:
    subset_df = df[df["label"] == label]
    subset = subset_df.sample(n=min(n_samples, len(subset_df)), random_state=random_state)

    all_pixels: list[np.ndarray] = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Building reference"):
        img_path = row.get(image_column, None)
        if pd.isna(img_path):
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        all_pixels.append(image.reshape(-1))

    if not all_pixels:
        raise ValueError("No readable images available to build reference histogram")

    stacked = np.concatenate(all_pixels)
    hist, _ = np.histogram(stacked, bins=256, range=(0, 256), density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    return cdf


def standardize_intensity(
    image: np.ndarray,
    target_mean: float = TARGET_MEAN,
    target_std: float = TARGET_STD,
) -> np.ndarray:
    image_f = image.astype(np.float32)
    body_mask = image_f > 10
    if int(body_mask.sum()) < 100:
        return image_f.astype(np.uint8)

    body_pixels = image_f[body_mask]
    image_mean = float(body_pixels.mean())
    image_std = max(float(body_pixels.std()), 1.0)
    image_f[body_mask] = ((body_pixels - image_mean) / image_std) * target_std + target_mean
    image_f[~body_mask] = 0
    return np.clip(image_f, 0, 255).astype(np.uint8)


def extreme_standardize(
    image: np.ndarray,
    target_mean: float = TARGET_MEAN,
    target_std: float = TARGET_STD,
) -> np.ndarray:
    image_f = image.astype(np.float32)
    body_mask = image_f > 5
    if int(body_mask.sum()) < 100:
        return np.full_like(image_f, int(target_mean), dtype=np.uint8)

    body_pixels = image_f[body_mask]
    z_scores = (body_pixels - body_pixels.mean()) / (body_pixels.std() + 1e-8)
    image_f[body_mask] = z_scores * target_std + target_mean
    image_f[~body_mask] = 0
    return np.clip(image_f, 0, 255).astype(np.uint8)


def histogram_match(source_image: np.ndarray, reference_cdf: np.ndarray) -> np.ndarray:
    src_hist, _ = np.histogram(source_image.reshape(-1), bins=256, range=(0, 256), density=True)
    src_cdf = src_hist.cumsum()
    src_cdf = src_cdf / src_cdf[-1]

    lookup = np.zeros(256, dtype=np.uint8)
    for src_value in range(256):
        ref_value = np.searchsorted(reference_cdf, src_cdf[src_value])
        lookup[src_value] = min(int(ref_value), 255)
    return lookup[source_image]


def presave_standardized_images(
    df_split: pd.DataFrame,
    output_root: str | Path,
    image_column: str = "image_path",
) -> dict[str, int]:
    output_root = as_path(output_root)
    (output_root / "cancer").mkdir(parents=True, exist_ok=True)
    (output_root / "control").mkdir(parents=True, exist_ok=True)

    skipped = 0
    saved = 0
    missing = 0

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc="Standardizing images"):
        img_path = Path(str(row[image_column]))
        if not img_path.exists():
            missing += 1
            continue

        out_dir = output_root / str(row["label"]) / str(row["patient_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"slice_{int(row['slice_idx']):03d}.png"

        if out_path.exists():
            skipped += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            missing += 1
            continue

        standardized = extreme_standardize(image)
        cv2.imwrite(str(out_path), standardized)
        saved += 1

    return {"saved": saved, "skipped": skipped, "missing": missing}


def analyze_patient(patient_path: str | Path) -> dict[str, float]:
    files = sorted(path for path in as_path(patient_path).iterdir() if path.is_file())
    if not files:
        return {
            "n_slices": 0,
            "mean_body_ratio": 0.0,
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "table_top_pct": 0.0,
            "table_bottom_pct": 0.0,
        }

    stats = {
        "body_ratios": [],
        "means": [],
        "stds": [],
        "table_top": [],
        "table_bottom": [],
    }

    sample_indices = np.linspace(0, len(files) - 1, min(5, len(files)), dtype=int)
    for index in sample_indices:
        image = cv2.imread(str(files[index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        body_mask = image > 10
        body_ratio = float(np.sum(body_mask) / image.size)
        stats["body_ratios"].append(body_ratio)

        if np.any(body_mask):
            body_pixels = image[body_mask]
            stats["means"].append(float(np.mean(body_pixels)))
            stats["stds"].append(float(np.std(body_pixels)))

        top_band = image[: int(image.shape[0] * 0.15), :]
        bottom_band = image[int(image.shape[0] * 0.85) :, :]
        stats["table_top"].append(float(np.mean(top_band > 10)))
        stats["table_bottom"].append(float(np.mean(bottom_band > 10)))

    return {
        "n_slices": len(files),
        "mean_body_ratio": float(np.mean(stats["body_ratios"])) if stats["body_ratios"] else 0.0,
        "mean_intensity": float(np.mean(stats["means"])) if stats["means"] else 0.0,
        "std_intensity": float(np.mean(stats["stds"])) if stats["stds"] else 0.0,
        "table_top_pct": float(np.mean(stats["table_top"])) if stats["table_top"] else 0.0,
        "table_bottom_pct": float(np.mean(stats["table_bottom"])) if stats["table_bottom"] else 0.0,
    }


def summarize_standardized_patient_dirs(root: str | Path) -> pd.DataFrame:
    root = as_path(root)
    records: list[dict[str, float | str]] = []
    for label_dir in [root / "cancer", root / "control"]:
        if not label_dir.exists():
            continue
        for patient_dir in sorted(path for path in label_dir.iterdir() if path.is_dir()):
            record = analyze_patient(patient_dir)
            record["label"] = label_dir.name
            record["patient_id"] = patient_dir.name
            records.append(record)
    return pd.DataFrame(records)


def describe_output_location(path: str | Path) -> str:
    return fmt_path(as_path(path))
