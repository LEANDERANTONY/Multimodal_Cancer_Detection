# Model Card

This document summarizes the main modelling components represented in the repository.

## Project Scope

The repository studies pancreatic cancer detection from two modalities:

- CT imaging
- urinary biomarkers

It also includes exploratory multimodal fusion, but the fusion branch should not be treated as a clinically validated multimodal system because the cohorts are not patient-paired.

## CT Model

### Model

- architecture: ResNet50-based classifier
- input: processed CT slice images after bias-aware preprocessing, orientation correction, segmentation, and cropping
- task: cancer vs control classification

### Strengths

- very strong tracked performance in the current repository outputs
- supported by a more mature preprocessing and analysis pipeline than the earlier notebook-only versions

### Limitations

- residual domain structure may still be present in learned features
- strong performance may partially reflect shortcut signal rather than purely pathological signal
- evaluation is still research-stage, not deployment-grade external validation

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

- dependency manager: `uv`
- main orchestration surface: `notebooks/01_multimodal_cancer_detection.ipynb`
- reusable implementation surface: `src/`
- tracked outputs: `reports/` and `figures/`
- local-only assets: `data/`, `models/`, `embeddings/`, `thesis/`

## Current Best Reading Of Results

- biomarker branch: strongest defensible standalone result
- CT branch: promising but scientifically ambiguous because of residual bias risk
- fusion branch: exploratory and hypothesis-generating rather than clinically validated

## Future Model Directions

- domain-adversarial CT training with a gradient reversal layer
- stronger external validation
- uncertainty and calibration improvements
- true paired multimodal evaluation if paired cohorts become available
