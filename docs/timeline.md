# Project Timeline

This document now serves as a historical-and-current timeline for the repository.

It is not a forward-looking daily execution plan anymore. For current priorities, use:

- `ROADMAP.md`
- `DEVLOG.md`
- `project_strategy.md`

## 2025 Thesis Execution Phase

The project originally ran as a notebook-first thesis workflow centered on:

- CT classification experiments
- urinary biomarker modelling
- exploratory multimodal fusion
- thesis figures, tables, and presentation outputs

Important practical constraints during that phase:

- CT and biomarker datasets were not patient-paired
- CT-heavy work depended on Google Colab GPU usage
- much of the implementation lived in evolving notebooks and local folders rather than a stable GitHub-tracked repo

That phase produced the core experiment logic, many of the tracked figures and reports, and the thesis-oriented analysis direction that still shapes the repository.

## Early 2026 Repository Consolidation Phase

The next major phase was repository stabilization.

Key outcomes:

- the GitHub-tracked repo under `Documents/Projects/Multimodal_Cancer_Detection` became the long-term working repo
- the latest practical notebook, reports, and figures were merged into that repo
- the modular scaffold under `src/`, `configs/`, `docs/`, and `results/` was preserved
- stale/generated folders were removed from the tracked working tree
- `.gitignore` was tightened so raw data, processed assets, models, embeddings, thesis files, and similar heavy local files remain out of Git

## 2026 Modularization Phase

After consolidation, the active focus shifted from merging files to improving structure.

Major milestones:

- migration from `requirements.txt` to `uv`
- clean `.venv` rebuild from `pyproject.toml` and `uv.lock`
- extraction of reusable helpers into:
  - `src/utils/`
  - `src/data/`
  - `src/models/`
  - `src/fusion/`
  - `src/interpretability/`
  - `src/results/`
- notebook refactors so the main notebook imports shared code for core CT, biomarker, fusion, and reporting sections

## Current Phase

The project is currently in a hybrid but much healthier state:

- the notebook remains the main research narrative
- stable reusable logic is moving into `src/`
- tracked reports and figures capture the latest lightweight outputs
- heavy assets remain local-only

Current emphasis:

- finish the notebook-to-module refactor for remaining exploratory helpers
- improve reproducibility and test coverage
- keep docs aligned with the actual repo state

## Historical Note

Some older planning assumptions no longer reflect the maintained repository exactly, especially:

- future-facing ideas that were never fully implemented
- infrastructure concepts like deployment-oriented APIs
- day-by-day scheduling targets from the thesis execution window

Those historical ideas still matter as context, but they should not be treated as the current source of truth for the repo.
