# DEVLOG - Multimodal Pancreatic Cancer Detection

This document tracks notable repository and implementation milestones.

Historical note:

- earlier work happened primarily in notebook-first local copies
- the current entries focus on the GitHub-tracked repository state after consolidation into the working repo under `Documents/Projects`

## Phase 0: Pre-Repository Thesis Execution (2025)

Before the tracked repository existed, the project ran as a notebook-first thesis workflow covering
CT classification, urinary biomarker modelling, exploratory multimodal fusion, and thesis figure /
table / presentation outputs.

Practical constraints during that phase:

- CT and biomarker datasets were not patient-paired
- CT-heavy work depended on Google Colab GPU usage
- most implementation lived in evolving notebooks and local folders rather than a stable repo

That phase produced the core experiment logic, many of the tracked figures and reports, and the
thesis-oriented analysis direction that still shapes the repository.

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

## Phase 11: PANORAMA External-Validation Setup (2026-07 to 2026-09)

Shifted from repo-hardening into the Q1 external-validation build.

- Acquired PANORAMA (largest public PDAC-detection CT cohort): all 4 Zenodo image batches (~193 GB) downloaded + MD5-verified, extracted (2,238 studies / 2,224 patients), reconciled 1:1 with the separate `panorama_labels` masks + `clinical_information.xlsx` (see `data/raw/ct/panorama/INVENTORY.md`).
- Ran a metadata confound audit (`reports/panorama_confound_audit.md`): scanner/manufacturer is a measured confound (Cramer's V = 0.44) and the `level` column is near-label leakage; there is no institution column, so the generalization axis becomes leave-one-manufacturer-out.
- Replanned the publication track to Q1-direct and reframed the CT contribution from "propose gradient reversal" to a comparative evaluation of mitigations (tuned ERM / DFR / GRL / SSL). See ADR-003.
- Consolidated docs (merged the former `docs/timeline.md` into this DEVLOG; centralized forward-looking plans in ROADMAP).
- Modernized the dependency stack (numpy 2.x, pandas 3.x, scikit-learn 1.9, ...) and moved torch to 2.9.1+cu128 so the RTX 5060 Ti (Blackwell / sm_120) is usable; Linux CI still resolves CPU torch from PyPI.
- Locked the v2 build decisions and a hybrid compute plan (3D nnU-Net reference on a rented RTX 4090; 2.5D mitigation panel local). See ADR-003 and `docs/preprocessing_audit.md` §5.
- Built + validated the pancreas ROI-crop pipeline (TotalSegmentator in an isolated env; pancreas bbox + a validated 150x100x40 mm margin gives 100% lesion + pancreas containment) and launched the full 2,238-case crop to `data/processed/ct/panorama_roi/`. Operational note: deep-dependency tool envs (torch) must live at SHORT paths on Windows or they hit the 260-char MAX_PATH limit and corrupt silently.

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

## Historical Caveats

Some older planning assumptions no longer reflect the maintained repository, especially:

- future-facing ideas that were never fully implemented
- infrastructure concepts such as deployment-oriented APIs
- day-by-day scheduling targets from the thesis execution window

Those historical ideas still matter as context, but they are not the current source of truth for
the repo. For current priorities use `ROADMAP.md`; for scope and rationale use `project_strategy.md`.
