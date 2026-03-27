from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm.auto import tqdm

from src.utils.project import as_path, fmt_path, require_cols, require_nonempty, require_path


def filter_cluster_outliers(
    metadata_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    output_metadata_path: str | Path,
    cluster_column: str = "cluster_k6",
    outlier_label: str = "cancer",
    outlier_cluster: int = 0,
) -> pd.DataFrame:
    metadata_path = require_path(metadata_path, "cluster metadata")
    input_root = require_path(input_root, "input CT root")
    output_root = as_path(output_root)
    output_metadata_path = as_path(output_metadata_path)

    df = pd.read_csv(metadata_path)
    require_cols(df, ["label", "patient", "file", cluster_column], "cluster metadata")
    require_nonempty(df, "cluster metadata")

    df_clean = df[~((df["label"] == outlier_label) & (df[cluster_column] == outlier_cluster))].copy()
    df_clean.to_csv(output_metadata_path, index=False)

    if output_root.exists():
        shutil.rmtree(output_root)
    for label in sorted(df_clean["label"].unique()):
        (output_root / label).mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Copying filtered slices"):
        src = input_root / str(row["label"]) / str(row["patient"]) / str(row["file"])
        if not src.exists():
            continue
        dst_dir = output_root / str(row["label"]) / str(row["patient"])
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / str(row["file"]))

    return df_clean


def orient_control_slices(
    input_root: str | Path,
    output_root: str | Path,
    log_path: str | Path,
) -> pd.DataFrame:
    input_root = require_path(input_root, "filtered CT root")
    output_root = as_path(output_root)
    log_path = as_path(log_path)

    if output_root.exists():
        shutil.rmtree(output_root)
    for label in ["cancer", "control"]:
        (output_root / label).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | bool]] = []
    for label in ["cancer", "control"]:
        label_dir = input_root / label
        if not label_dir.exists():
            continue
        patients = sorted(path for path in label_dir.iterdir() if path.is_dir())
        for patient_dir in tqdm(patients, desc=f"Orienting {label}"):
            out_dir = output_root / label / patient_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(path for path in patient_dir.iterdir() if path.is_file()):
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                flipped = False
                if label == "control":
                    image = cv2.rotate(image, cv2.ROTATE_180)
                    flipped = True

                cv2.imwrite(str(out_dir / image_path.name), image)
                records.append(
                    {
                        "label": label,
                        "patient": patient_dir.name,
                        "file": image_path.name,
                        "flipped": flipped,
                    }
                )

    df = pd.DataFrame(records)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_path, index=False)
    return df


def ct_body_mask(image: np.ndarray) -> np.ndarray:
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(image_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(image_norm[mask == 255]) < np.mean(image_norm[mask == 0]):
        mask = 255 - mask

    height, width = mask.shape
    center = np.zeros_like(mask)
    cv2.ellipse(center, (width // 2, height // 2), (int(0.45 * width), int(0.45 * height)), 0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, center)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = ndimage.binary_fill_holes(mask // 255).astype(np.uint8) * 255

    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_components > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255
    return mask


def segment_body(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ct_body_mask(image)
    return mask, image * (mask // 255)


def segment_oriented_dataset(
    input_root: str | Path,
    output_root: str | Path,
    stats_csv_path: str | Path,
) -> pd.DataFrame:
    input_root = require_path(input_root, "oriented CT root")
    output_root = as_path(output_root)
    stats_csv_path = as_path(stats_csv_path)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | float]] = []
    for label in ["cancer", "control"]:
        label_dir = input_root / label
        if not label_dir.exists():
            continue
        patients = sorted(path for path in label_dir.iterdir() if path.is_dir())
        for patient_dir in tqdm(patients, desc=f"Segmenting {label}"):
            out_dir = output_root / label / patient_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(path for path in patient_dir.iterdir() if path.is_file()):
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                mask, masked = segment_body(image)
                body_ratio = float(np.sum(mask > 0) / mask.size)
                cv2.imwrite(str(out_dir / image_path.name), masked)
                records.append(
                    {
                        "label": label,
                        "patient": patient_dir.name,
                        "file": image_path.name,
                        "body_ratio": body_ratio,
                    }
                )

    df = pd.DataFrame(records)
    stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stats_csv_path, index=False)
    return df


def compute_body_geometry(
    clean_stats_csv: str | Path,
    segmented_root: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    clean_stats_csv = require_path(clean_stats_csv, "clean body stats")
    segmented_root = require_path(segmented_root, "segmented CT root")
    output_csv = as_path(output_csv)

    df = pd.read_csv(clean_stats_csv)
    require_cols(df, ["label", "patient", "file"], "clean body stats")
    require_nonempty(df, "clean body stats")

    records: list[dict[str, str | int | float]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Body geometry"):
        image_path = segmented_root / str(row["label"]) / str(row["patient"]) / str(row["file"])
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        mask = ct_body_mask(image)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        left = int(xs.min())
        right = int(xs.max()) + 1
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        width = right - left
        height = bottom - top
        area = int(np.sum(mask > 0))
        image_area = int(image.shape[0] * image.shape[1])

        records.append(
            {
                "label": row["label"],
                "patient": row["patient"],
                "file": row["file"],
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "width": width,
                "height": height,
                "aspect": (width / height) if height else np.nan,
                "body_ratio": (area / image_area) if image_area else np.nan,
            }
        )

    df_geom = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_geom.to_csv(output_csv, index=False)
    return df_geom


def crop_using_geom(image: np.ndarray, geom_row: pd.Series | dict[str, int], size: tuple[int, int] = (224, 224)) -> np.ndarray:
    top = int(geom_row["top"])
    bottom = int(geom_row["bottom"])
    left = int(geom_row["left"])
    right = int(geom_row["right"])

    top = max(0, min(top, image.shape[0] - 1))
    bottom = max(top + 1, min(bottom, image.shape[0]))
    left = max(0, min(left, image.shape[1] - 1))
    right = max(left + 1, min(right, image.shape[1]))

    crop = image[top:bottom, left:right]
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def crop_segmented_dataset(
    segmented_root: str | Path,
    body_geometry_csv: str | Path,
    output_root: str | Path,
    metadata_output_csv: str | Path,
) -> pd.DataFrame:
    segmented_root = require_path(segmented_root, "segmented CT root")
    body_geometry_csv = require_path(body_geometry_csv, "body geometry")
    output_root = as_path(output_root)
    metadata_output_csv = as_path(metadata_output_csv)

    df_body = pd.read_csv(body_geometry_csv)
    require_cols(df_body, ["label", "patient", "file", "top", "bottom", "left", "right"], "body geometry")
    require_nonempty(df_body, "body geometry")
    df_body = df_body.set_index(["label", "patient", "file"])

    if output_root.exists():
        shutil.rmtree(output_root)
    for label in ["cancer", "control"]:
        (output_root / label).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for label in ["cancer", "control"]:
        label_dir = segmented_root / label
        if not label_dir.exists():
            continue
        patients = sorted(path for path in label_dir.iterdir() if path.is_dir())
        for patient_dir in tqdm(patients, desc=f"Cropping {label}"):
            out_dir = output_root / label / patient_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(path for path in patient_dir.iterdir() if path.is_file()):
                key = (label, patient_dir.name, image_path.name)
                if key not in df_body.index:
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                crop = crop_using_geom(image, df_body.loc[key])
                cv2.imwrite(str(out_dir / image_path.name), crop)
                records.append({"label": label, "patient": patient_dir.name, "file": image_path.name})

    df_meta = pd.DataFrame(records)
    metadata_output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_meta.to_csv(metadata_output_csv, index=False)
    return df_meta


def output_message(path: str | Path) -> str:
    return fmt_path(as_path(path))
