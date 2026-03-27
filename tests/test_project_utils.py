from pathlib import Path

import pandas as pd
import pytest

from src.utils.project import (
    build_project_paths,
    fmt_path,
    require_cols,
    require_nonempty,
    require_path,
    resolve_project_relative,
)


def test_build_project_paths_uses_given_root(tmp_path: Path) -> None:
    paths = build_project_paths(tmp_path)
    assert paths.project_root == tmp_path.resolve()
    assert paths.data_root == tmp_path / "data"
    assert paths.reports_dir == tmp_path / "reports"


def test_fmt_path_normalizes_backslashes() -> None:
    assert fmt_path(r"a\b\c.txt") == "a/b/c.txt"


def test_require_path_returns_existing_path(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("ok", encoding="utf-8")
    assert require_path(file_path) == file_path


def test_require_path_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        require_path(tmp_path / "missing.txt", "Missing file")


def test_require_cols_validates_missing_columns() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="missing columns"):
        require_cols(df, ["a", "c"], label="demo")


def test_require_nonempty_raises_for_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="empty"):
        require_nonempty(pd.DataFrame(), label="demo")


def test_resolve_project_relative_preserves_absolute_path(tmp_path: Path) -> None:
    absolute = tmp_path / "file.txt"
    assert resolve_project_relative(tmp_path, absolute) == absolute


def test_resolve_project_relative_joins_relative_path(tmp_path: Path) -> None:
    resolved = resolve_project_relative(tmp_path, "reports/out.csv")
    assert resolved == tmp_path / "reports" / "out.csv"
