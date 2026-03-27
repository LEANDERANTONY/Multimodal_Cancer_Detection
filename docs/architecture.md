# Architecture Overview

This document describes the current runtime architecture of the multimodal pancreatic cancer detection project.

## System Goal

The project supports a research workflow for pancreatic cancer detection using:

- CT imaging experiments
- urinary biomarker modelling
- exploratory multimodal fusion across those two modalities
- tracked reports and figures for thesis and research outputs

The current repository is a hybrid research codebase:

- `src/` holds reusable preprocessing, modelling, fusion, and reporting helpers
- `notebooks/01_multimodal_cancer_detection.ipynb` remains the primary orchestration and analysis surface
- `reports/` and `figures/` store curated tracked outputs
- raw data, processed datasets, embeddings, and model weights stay local-only

## High-Level Flow

1. Local datasets are placed under the ignored `data/raw/` tree.
2. Preprocessing scripts or notebook steps prepare CT and biomarker inputs.
3. The CT path performs bias detection, body-masked extreme standardization, orientation correction, body segmentation, and cropping before model training.
4. The notebook imports shared helpers from `src/` for paths, reproducibility, CT preparation, model training, and fusion.
5. CT experiments produce slice-level and patient-level metrics plus intermediate embeddings.
6. Biomarker experiments produce MLP-based classification outputs and diagnostics.
7. Fusion sections compare feature-level and decision-level strategies using synthetic pairing across separate cohorts, including negative controls.
8. Summary helpers export final JSON, CSV, and figure artifacts to `reports/` and `figures/`.

Current interpretation of those stages:

- CT performance is strong but still partially ambiguous because residual domain structure may persist in deep features
- biomarker performance is the cleanest reproducible positive signal in the repo
- fusion infrastructure is methodologically useful, but current fusion outputs must remain exploratory due to synthetic pairing

## Main Modules

### `notebooks/01_multimodal_cancer_detection.ipynb`

This is still the main end-to-end research notebook.

Its current role is:

- orchestrating the full workflow
- running exploratory analyses
- generating reports and figures
- importing reusable logic from `src/` instead of redefining core implementations inline

The notebook is still the most complete experiment narrative, but it is no longer the only implementation surface.

### `src/utils/`

Owns project-level utilities:

- project-root and path resolution
- formatting helpers
- reproducibility helpers such as seeding
- lightweight logging

Key files:

- `src/utils/project.py`
- `src/utils/repro.py`

### `src/data/`

Owns CT-oriented data analysis and preprocessing helpers:

- dataset and path inspection
- histogram and intensity standardization helpers
- CT bias and slice analysis helpers
- cropping, orientation, segmentation, and body-geometry pipeline helpers
- CT embedding extraction and clustering support

This reflects the actual implemented dissertation pipeline more accurately than an ROI-detector-first description.

Tracked preprocessing scripts also live under:

- `src/data/preprocess/ct_preprocess.py`
- `src/data/preprocess/biomarker_preprocess.py`

### `src/models/`

Owns reusable model-layer code for:

- CT slice datasets and dataloaders
- CT model construction and training loops
- checkpoint loading
- cluster-specific CT reporting helpers
- biomarker MLP modelling and evaluation

The current CT stack is centered around a ResNet50 classifier, while the biomarker stack uses an MLP.

### `src/fusion/`

Owns exploratory multimodal fusion helpers.

Current reusable coverage includes:

- feature-level embedding concatenation
- decision-level weighted fusion
- synthetic label-matched and mismatch-control pairing utilities
- multi-seed sanity checks for fusion analysis

Important constraint:

- fusion experiments are exploratory because CT and biomarker data are not patient-paired in the current datasets

### `src/interpretability/`

Owns Grad-CAM logic used for qualitative CT inspection.

### `src/results/`

Owns report assembly helpers for:

- loading current or saved metrics
- building final summary JSON payloads
- building model comparison tables

## State Model

This is a file-oriented research project rather than a long-running application.

The main forms of state are:

- notebook kernel state during active experimentation
- local ignored assets under `data/`, `models/`, and `embeddings/`
- tracked lightweight artifacts in `reports/` and `figures/`

That split is deliberate:

- heavy data and checkpoints remain local
- code, docs, curated outputs, and lightweight summaries remain versioned

## Dependency and Environment Model

The repository uses `uv` as the dependency and environment source of truth.

Authoritative files:

- `pyproject.toml`
- `uv.lock`
- `.python-version`

Typical workflow:

- `uv sync`
- `uv run python ...`

`requirements.txt` is no longer part of the maintained workflow.

## Reproducibility Model

The current reproducibility approach is lightweight and practical rather than fully automated:

- shared seed helpers in `src/utils/repro.py`
- shared path resolution in `src/utils/project.py`
- tracked reports and figures for important outputs
- notebook-driven orchestration for the latest experiment flow

Current limitations:

- the automated `tests/` suite is currently lightweight and focused on stable helper modules rather than the full end-to-end workflow
- the full notebook is not yet covered by automated smoke tests
- reproducibility still depends on local dataset availability and local-only assets

## Current Constraints

- the repo is still consolidating from a notebook-first research workflow into a more modular structure
- some exploratory helpers remain inline in the notebook
- there is no packaged CLI or service runtime yet
- fusion results rely on synthetic pairing across separate cohorts
- trained checkpoints and processed datasets are intentionally not tracked in Git

## Next Architecture Step

The next meaningful step is not backend extraction or deployment. It is research-code hardening inside the current repository shape.

Near-term targets:

- continue moving reusable notebook logic into `src/`
- reduce duplication in exploratory sections that still live inline in the notebook
- add lightweight automated tests for reusable module layers
- tighten report-generation and reproducibility guidance in docs
- preserve the implemented bias-aware CT pipeline description consistently across repo docs

Later targets:

- add explicit runner scripts or thin CLIs for common experiment/report tasks
- separate stable pipeline helpers from one-off exploratory notebook logic even more cleanly
- explore domain-adversarial CT training via a gradient reversal layer to reduce residual dataset-of-origin information in learned features
