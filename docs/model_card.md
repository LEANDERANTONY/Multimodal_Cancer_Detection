# Model Card

This document summarizes the main modelling components represented in the repository.

## Project Scope

The repository studies pancreatic cancer detection from two modalities:

- CT imaging
- urinary biomarkers

It also includes exploratory multimodal fusion, but the fusion branch should not be treated as a clinically validated multimodal system because the cohorts are not patient-paired.

## PANORAMA 3D nnU-Net Models (Q1 work, 2026)

### Model

- architecture: nnU-Net v2 3d_fullres (PlainConvUNet, 6 stages), `nnUNetTrainer_250epochs`, CT normalisation (clip + z-score)
- input: pancreas-region crops of contrast-enhanced CT, cut from PANORAMA's provided pancreas + duct masks (never the lesion mask); **tight** 100×50×15 mm margin (primary) and **loose** 150×100×40 mm (ablation)
- task: PDAC lesion segmentation; patient-level detection score from the softmax (p_max, p_sum, PanDx-style cc_psz)
- training data: 1,964 Dutch PANORAMA scans; 194 MSD + 80 NIH held out as external tests
- checkpoints: local `models/nnunet/{loose_cv, loose_lomo, tight_cv, tight_lomo}/` (not tracked)

### Performance (tight crop)

| Setting | Lesion Dice (PDAC cases) | Detection AUROC |
|---|---|---|
| 5-fold CV | 0.505 | 0.787 (cc_psz) |
| LOMO (Siemens / Toshiba / Philips held out) | 0.547 / 0.450 / 0.471 | 0.811 / 0.773 / 0.706 |
| External MSD (new hospital, 5-fold ensemble) | 0.555 | 0.823 [0.764, 0.884] (p_max) |

Confound tax (in-distribution): +0.031 AUROC [+0.010, +0.053]. Loose crop: CV Dice ~0.33, detection ~0.70.

### Limitations

- trained and tested on crops from **provided** masks, unavailable clinically — deployment with a predicted pancreas mask is not yet measured (ADR-005)
- a small, measurable share of performance comes from the scanner/case-mix confound (bounded ≈0.05 AUROC)
- decision thresholds set on the Dutch CV scores transfer poorly to the external ensemble scores (operating-point check pending)
- a single-stage reference model, below the PANORAMA challenge winner (PanDx, AUROC 0.926)
- one LOMO training run collapsed to all-background and was re-run (reported)

## Thesis-Era CT Model (2025)

### Model

- architecture: ResNet50-based classifier
- input: processed CT slice images after bias-aware preprocessing, orientation correction, segmentation, and cropping
- task: cancer vs control classification

### Interpretation

- the tracked AUC (0.9999 slice / 1.000 patient) is a **dataset-of-origin artifact**: cancer scans came from one source dataset and controls from another, so separating the datasets separates the classes
- ResNet50 embeddings still cluster by source after pixel standardisation
- kept as the motivating example (positive control) for the Q1 work, never as a performance claim

## Biomarker Model

### Model

- architecture: MLP
- input features:
  - `age`
  - `plasma_CA19_9`
  - `creatinine`
  - `LYVE1`
  - `REG1B`
  - `TFF1`
  - `REG1A`
- task: cancer vs non-cancer classification

### Strengths

- this is the clearest reproducible positive result in the repository
- uses a compact and interpretable feature set relative to the CT branch

### Limitations

- still evaluated as a research model
- performance should not be generalized beyond the study setting without additional external validation

## Fusion Models

### Covered Strategies

- decision-level weighted fusion
- feature-level embedding fusion
- label-matched and label-mismatch sanity controls

### Interpretation

- fusion experiments are methodologically useful
- current fusion results are exploratory only
- they should not be described as evidence of true multimodal clinical benefit because the modalities are not patient-paired

## Training And Runtime Context

- dependency manager: `uv` (project `.venv`); nnU-Net runs in a separate env, `data/envs/nnunet` (torch 2.8)
- PANORAMA models: `scripts/runpod/` + `tools/` (see `docs/modeling_pipeline.md`)
- thesis orchestration surface: `notebooks/01_multimodal_cancer_detection.ipynb`
- reusable implementation surface: `src/`
- tracked outputs: `reports/` and `figures/`
- local-only assets: `data/`, `models/`, `embeddings/`, `thesis/`

## Current Best Reading Of Results

- biomarker branch: strongest defensible standalone result
- PANORAMA 3D CT models: generalise across manufacturers and to an external hospital; the scanner confound is only weakly exploited
- thesis CT model: confounded, kept as the motivating example
- fusion branch: exploratory and hypothesis-generating rather than clinically validated

## Future Model Directions

- deployment-ROI evaluation with predicted pancreas masks (ADR-005)
- mitigation panel on the 2.5D backbone: tuned ERM, group-balanced, DFR, GRL, MAE pretraining
- calibration (ECE/Brier) and uncertainty
- true paired multimodal evaluation if paired cohorts become available
