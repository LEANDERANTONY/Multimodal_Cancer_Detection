# Project Strategy

This document captures the current project and repository direction for the multimodal pancreatic cancer detection study.

> Companion docs: `ROADMAP.md` holds the forward-looking publication plan (Q2 shortcut-learning
> paper now, Q1 PANORAMA external-validation + domain-adversarial track next). `docs/preprocessing_audit.md`
> holds the full CT preprocessing audit and the v2 (PANORAMA) reprocessing spec. This file covers the
> *why* and the scope/structure rationale; the ROADMAP covers *what's next*.

## Current Project Position

This repository is currently a research-first codebase rather than a deployable product.

Its active purpose is to support:

- CT imaging experiments
- urinary biomarker modelling
- exploratory multimodal fusion analysis
- report, figure, and thesis-oriented output generation

The current scientific stance of the repo should mirror the dissertation:

- the biomarker branch is the clearest positive result and strongest reproducibility story
- the CT branch is promising but still scientifically ambiguous because residual domain structure remains a live concern
- the fusion branch is valuable mainly as a carefully controlled exploratory framework, not as evidence of proven multimodal clinical benefit

That means the docs should describe the implemented CT pipeline as bias-aware preprocessing plus ResNet50 classification, not as a YOLO-based detector pipeline.

The repository is no longer just a dump of notebooks and artifacts, but it is also not yet a fully script-driven research pipeline. It sits deliberately in the middle:

- reusable code in `src/`
- orchestration and narrative still centered in one main notebook

## Scope Boundaries

The active project scope is intentionally narrow:

- CT modality
- urinary biomarkers
- fusion methodology exploration across those modalities

What that means in practice:

- no blood biomarker branch in the current maintained workflow
- no claim of clinically validated multimodal performance on paired patients
- no deployment API, frontend, or application runtime
- no commitment to tracking large raw or derived assets in Git

## Repo Strategy

The repository should stay opinionated about what is tracked.

Tracked:

- source code
- lightweight configs
- current-state docs
- curated reports
- curated figures
- the main experiment notebook

Local-only:

- raw data
- processed datasets
- embeddings
- model checkpoints
- thesis working materials
- large archives and temporary staging folders

That split keeps the repo shareable without pretending to be a full data lake.

## Notebook Strategy

The notebook is still important and should stay.

Its role should be:

- experiment orchestration
- visual analysis
- diagnostic exploration
- final report assembly

Its role should not be:

- the only place core training logic exists
- the only place path handling exists
- the only place fusion logic exists

The ongoing strategy is to keep moving stable logic into `src/` while preserving the notebook as the main research narrative.

## Code Organization Strategy

The active module boundaries are now good enough to build on:

- `src/utils/` for project infrastructure
- `src/data/` for preprocessing, CT analysis, and embeddings
- `src/models/` for CT and biomarker model logic
- `src/fusion/` for multimodal fusion helpers
- `src/interpretability/` for Grad-CAM
- `src/results/` for summary/report assembly

The next cleanup priority is depth, not breadth:

- improve the existing module boundaries
- reduce remaining notebook-only duplication
- avoid inventing extra architecture layers before they are needed

## Reproducibility Strategy

The current reproducibility model should stay practical:

- use `uv` as the dependency source of truth
- use stable path and seeding helpers
- keep important outputs in tracked `reports/` and `figures/`
- add targeted automated tests for reusable helpers over time

What we should not do yet:

- over-engineer a heavy workflow engine
- force every exploratory notebook step into rigid script interfaces prematurely

## Fusion Strategy

Fusion work must stay clearly labeled as exploratory.

That is because:

- CT and biomarker cohorts are not patient-paired in the current datasets
- decision-level and feature-level fusion rely on synthetic matching assumptions

The project should keep treating fusion as:

- a methodological comparison
- a future-facing design exercise
- a source of research insight
- a negative-control-sensitive evaluation problem

not as a final clinical performance claim.

## What We Should Keep Doing

- keep extracting reusable notebook logic into `src/`
- keep docs aligned with the current repo state
- keep `uv` and imports honest
- keep tracked artifacts curated rather than dumping every intermediate output into Git
- keep the repo understandable to a new collaborator without access to the private raw data
- keep negative findings visible in docs rather than smoothing them away into optimistic summaries

## What We Should Not Add Yet

- a FastAPI service
- a frontend app
- containerization purely for the sake of having Docker
- heavy automation around experiments that are still rapidly changing

## Near-Term Priorities

The highest-value work from the current state is:

1. finish the notebook-to-module refactor for the remaining exploratory helpers
2. add lightweight tests for stable helper modules
3. improve documentation consistency and reproducibility guidance
4. make key outputs easier to regenerate without re-reading the whole notebook manually
5. keep the external-validation and matched-cohort story explicit in planning so future work addresses the real scientific bottlenecks
6. explicitly track domain-adversarial training with gradient reversal as a future bias-reduction direction for CT modelling

## Longer-Term Path

If the repo becomes more stable and publication-facing, the next sensible path is:

1. keep the current module boundaries stable
2. add thin script or CLI runners around common experiment/report tasks
3. improve test coverage around those stable entry points
4. only then consider broader packaging or deployment-oriented structure
