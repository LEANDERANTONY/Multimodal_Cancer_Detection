from __future__ import annotations

from pathlib import Path

import yaml

from .project import build_project_paths


def load_paths(cfg_path: str | Path = "configs/paths.yaml", make_dirs: bool = True) -> dict[str, str]:
    cfg_file = Path(cfg_path)
    with cfg_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    project = build_project_paths(cfg_file.resolve().parents[1])
    resolved: dict[str, str] = {}
    for key, value in config.items():
        path = Path(value)
        if not path.is_absolute():
            path = project.project_root / path
        if make_dirs:
            path.mkdir(parents=True, exist_ok=True)
        resolved[key] = str(path.resolve())
    return resolved
