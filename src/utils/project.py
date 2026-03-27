from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    processed_root: Path
    ct_images_dir: Path
    split_index: Path
    biomarkers_path: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    embeddings_dir: Path


def detect_project_root(env_var: str = "MCD_PROJ") -> Path:
    override = os.environ.get(env_var, "").strip()
    if override:
        return Path(override).expanduser().resolve()

    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents[:4]]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "reports").exists():
            return candidate
    return cwd


def build_project_paths(project_root: str | Path | None = None) -> ProjectPaths:
    root = Path(project_root).expanduser().resolve() if project_root else detect_project_root()
    data_root = root / "data"
    processed_root = data_root / "processed"
    processed_biomarkers = processed_root / "biomarkers_clean.csv"
    raw_biomarkers = data_root / "raw" / "biomarkers" / "urinary_biomarkers.csv"
    return ProjectPaths(
        project_root=root,
        data_root=data_root,
        processed_root=processed_root,
        ct_images_dir=processed_root / "ct_images",
        split_index=root / "reports" / "df_split_cropped.csv",
        biomarkers_path=processed_biomarkers if processed_biomarkers.exists() else raw_biomarkers,
        models_dir=root / "models",
        reports_dir=root / "reports",
        figures_dir=root / "figures",
        embeddings_dir=root / "embeddings",
    )


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def fmt_path(path: str | Path) -> str:
    return str(as_path(path)).replace("\\", "/")


def require_path(path: str | Path, label: str | None = None) -> Path:
    resolved = as_path(path)
    if not resolved.exists():
        name = label or resolved.name or str(resolved)
        raise FileNotFoundError(f"{name} not found: {fmt_path(resolved)}")
    return resolved


def require_cols(df: "pd.DataFrame", columns: Iterable[str], label: str = "DataFrame") -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def require_nonempty(df: "pd.DataFrame", label: str = "DataFrame") -> None:
    if df.empty:
        raise ValueError(f"{label} is empty")


def resolve_project_relative(project_root: str | Path, path_value: str | Path) -> Path:
    candidate = as_path(path_value)
    if candidate.is_absolute():
        return candidate
    return as_path(project_root) / candidate


def resolve_ct_dir(project: ProjectPaths, dirname: str) -> Path:
    return project.processed_root / dirname


def resolve_reports_dir(project: ProjectPaths) -> Path:
    return project.reports_dir


def resolve_figures_dir(project: ProjectPaths) -> Path:
    return project.figures_dir


def resolve_models_dir(project: ProjectPaths) -> Path:
    return project.models_dir


def resolve_embeddings_dir(project: ProjectPaths) -> Path:
    return project.embeddings_dir
