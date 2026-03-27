# DEVLOG - Multimodal Pancreatic Cancer Detection

This document tracks notable repository and implementation milestones.

Historical note:

- earlier work happened primarily in notebook-first local copies
- the current entries focus on the GitHub-tracked repository state after consolidation into the working repo under `Documents/Projects`

## Phase 1: Hybrid Repo Consolidation

- Confirmed the GitHub-tracked repository and treated it as the long-term source of truth.
- Preserved the modular scaffold under `src/`, `configs/`, `docs/`, and `results/`.
- Merged in the latest practical notebook, report, and figure assets from the newer working copy.
- Cleaned stale or generated folders before the merge snapshot.
- Updated `.gitignore` so raw data, processed data, embeddings, models, thesis assets, and similar local-only files stay out of Git.
- Verified the repo could now serve as the stable working directory despite earlier Google Drive filesystem issues.

## Phase 2: README And Repo Baseline Refresh

- Rewrote the root README so it reflects the actual hybrid repo state rather than the older aspirational structure.
- Normalized tracked report content that leaked local-path assumptions.
- Preserved the modular code areas while acknowledging that the notebook was still the primary orchestration surface.

## Phase 3: `uv` Migration

- Removed `requirements.txt` from the maintained workflow.
- Added:
  - `pyproject.toml`
  - `uv.lock`
  - `.python-version`
- Rebuilt `.venv` cleanly with `uv sync`.
- Updated setup instructions and command examples to use `uv run`.

## Phase 4: Shared Utility Extraction

- Added project/path utilities in `src/utils/project.py`.
- Added seeding helpers in `src/utils/repro.py`.
- Updated exports through `src/utils/__init__.py`.
- Shifted notebook bootstrap logic toward importing reusable helpers rather than redefining them inline.

## Phase 5: CT Pipeline Modularization

- Added CT analysis helpers in `src/data/ct_analysis.py`.
- Added CT preprocessing and image-shaping helpers in `src/data/ct_pipeline.py`.
- Added CT embedding and clustering helpers in `src/data/ct_embeddings.py`.
- Kept tracked preprocessing scripts in `src/data/preprocess/`.
- Rewired multiple notebook sections to use shared data-layer code.
- Aligned repo documentation with the actual implemented CT path from the dissertation: bias-aware preprocessing, segmentation/cropping, and ResNet50 classification rather than a YOLO-first detector pipeline.

## Phase 6: Model And Interpretability Modularization

- Added CT dataset, model, evaluation, and training helpers in `src/models/ct.py`.
- Added cluster-specific CT result helpers in `src/models/clustered_ct.py`.
- Added biomarker modelling helpers in `src/models/biomarker.py`.
- Added Grad-CAM support in `src/interpretability/gradcam.py`.
- Rewired core CT training, evaluation, and interpretability notebook sections to use `src`.

## Phase 7: Fusion And Results Modularization

- Added feature-level fusion helpers in `src/fusion/feature_level.py`.
- Added decision-level fusion helpers in `src/fusion/decision_level.py`.
- Added final summary and model-comparison helpers in `src/results/summary.py`.
- Reworked the notebook's fusion and final reporting tail so those sections call shared modules instead of defining logic inline.
- Kept the fusion framing aligned with the dissertation outcome: methodologically useful, but not evidence of clinically validated multimodal synergy under synthetic pairing.

## Phase 8: Documentation Spine Alignment

- Added architecture, roadmap, devlog, strategy, and ADR index documents so this repo has a documentation backbone similar in quality to the sibling AI Job Application Agent project.
- Updated the README documentation section to point readers toward the new current-state docs.
- Re-read the final dissertation chapters so the repo docs reflect the thesis-level interpretation, not just the code layout.

## Phase 9: Lightweight Test Baseline

- Added an initial `pytest` suite under `tests/`.
- Covered stable helper layers for:
  - project/path utilities
  - decision-level fusion helpers
  - results summary/report helpers
- Added pytest configuration and dependency wiring through `pyproject.toml` and `uv`.
- Verified the suite with `uv run pytest`.

## Phase 10: Processed-Data-First Runtime Clarification

- Verified that the biomarker modelling flow is compatible with `data/processed/biomarkers_clean.csv`, not just the original raw CSV.
- Updated project path resolution so normal notebook runs prefer processed biomarker inputs and fall back to raw only for local rebuild scenarios.
- Aligned README guidance with the actual maintained workflow: notebook-first analysis runs should start from `data/processed/` for both CT and biomarkers.

## Current Verification Practice

The main validation steps currently used are:

- `uv run python -m compileall src`
- targeted import smoke tests for new module layers
- notebook JSON inspection after programmatic rewrites
- `uv run pytest`

## Current Gaps

- the current `tests/` suite is intentionally small and covers only stable helpers so far
- the full notebook still is not executed as an automated smoke test
- some earlier exploratory notebook sections still contain inline helper code that can be extracted later
- CT generalization remains scientifically ambiguous until stronger external validation is added
