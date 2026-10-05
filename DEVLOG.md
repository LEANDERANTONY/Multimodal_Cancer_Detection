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
- Built + validated the pancreas ROI-crop pipeline. TotalSegmentator (isolated env) was tested first but failed on ~5-14% of scans, so the crop uses PANORAMA's provided, label-blind pancreas (4) + duct (5) masks with a validated 150x100x40 mm margin (100% lesion + pancreas containment); the full 2,238-case "mask-based crop" ran to `data/processed/ct/panorama_roi/` (since superseded by Dataset700, see Phase 12). Operational note: deep-dependency tool envs (torch) must live at SHORT paths on Windows or they hit the 260-char MAX_PATH limit and corrupt silently.

## Phase 12: 3D nnU-Net Reference, Loose Crop (2026-09-27 to 2026-10-02)

- Converted the crops to nnU-Net format as **Dataset700** (1,964 Dutch train + 274 MSD/NIH held out; image origins restored — the first crop pass had zeroed them).
- Trained nnU-Net v2 3d_fullres (250 epochs) on rented RTX 3090/4090 pods: 5-fold CV (lesion Dice ~0.33) and LOMO with Siemens / Toshiba / Philips held out (Dice 0.365 / 0.349 / 0.287).
- Detection scoring from the softmax (p_max, p_sum, PanDx-style cc_psz): CV AUROC ~0.70; per-scanner LOMO detection 0.64 / 0.71 / 0.59.
- Scanner-metadata-only baseline AUROC 0.70 (`tools/scanner_only_shortcut.py`); confound tax ≈ +0.02 (no CI then); encoder feature probe: scanner 0.59-0.61 vs cancer 0.62-0.69. First reading: confound present, not substantially exploited.
- Addressed a Codex review of the confound-audit wording; wrote `docs/q1_readiness_and_gaps.md`.

## Phase 13: Tight Crop, Tight LOMO, Confound Battery (2026-10-02 to 2026-10-05)

- Built **Dataset701** with the field-standard tight margin 100x50x15 mm (`tools/build_roi_dataset.py`). Tight CV: Dice 0.505, detection cc_psz 0.787 — a clear win over loose.
- Tight LOMO: Dice 0.547 / 0.450 / 0.471, detection 0.811 / 0.773 / 0.706, matching in-distribution per-scanner values (no manufacturer-shift penalty).
- **Fold 2 (Philips held out) collapsed** to all-background on its first run (per-sample Dice rewards empty predictions; that fold has the lowest positive rate). Diagnosed (blank even on its own training scans), archived as `fold_2_collapsed/`, re-run successfully.
- Tight confound battery (`scripts/runpod/tight_battery.py`): confound tax **+0.031 [+0.010, +0.053]** (small but non-zero), feature probe scanner 0.62-0.64 vs cancer 0.75-0.77. Reframed the claim to "present, only weakly exploited (bounded)". Found a mislabel in earlier notes (loose-LOMO detection AUROCs had been recorded as Dice) and corrected it. See ADR-004.

## Phase 14: External Validation, Local Mirror, Pre-Registration (2026-10-05)

- External validation on the never-trained-on MSD (194, 98 PDAC) and NIH (80, all healthy) sets, 5-fold ensemble: tight MSD AUROC 0.82 [0.76, 0.88], Dice 0.555. MSD and NIH are reported separately (pooling would reintroduce a dataset-of-origin confound). Loose external: MSD AUROC 0.71, Dice 0.35 (its in-distribution level). Per-fold operating points: p_max thresholds set on Dutch CV transfer to MSD/NIH; cc_psz thresholds shift (lower MSD sensitivity), a score-scale effect, not ensemble smoothing.
- Mirrored every run from the RunPod volume locally (`models/nnunet/`, 16 models), reorganised local data into a documented layout (`docs/data_layout.md`), removed duplicate crops/tars, committed all run summaries and per-case scores to `reports/nnunet_summaries/`.
- Started a local loose-CV re-inference (`tools/loose_cv_reinfer.py`, separate torch-2.8 env because nnU-Net excludes torch 2.9) to put a CI on the loose confound tax.
- Pre-registered the deployment-ROI experiment and the mitigation panel (`docs/deployment_and_mitigation_design.md`, ADR-005), borrowing practices from studied Kaggle grandmasters.
- Documentation pass: docs index (`docs/README.md`), README status, ADR-004/005, refreshed architecture, model card and roadmap.

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
- the PANORAMA pipeline lives in scripts (`scripts/runpod/`, `tools/`) rather than `src/`, and has no automated tests yet
- open Q1 items: loose tax CI, deployment-ROI, mitigation panel (see `docs/q1_readiness_and_gaps.md`)

## Historical Caveats

Some older planning assumptions no longer reflect the maintained repository, especially:

- future-facing ideas that were never fully implemented
- infrastructure concepts such as deployment-oriented APIs
- day-by-day scheduling targets from the thesis execution window

Those historical ideas still matter as context, but they are not the current source of truth for
the repo. For current priorities use `ROADMAP.md`; for scope and rationale use `docs/project_strategy.md`.
