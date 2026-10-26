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
| 5-fold CV | 0.507 | 0.787 (cc_psz) |
| LOMO (Siemens / Toshiba / Philips held out) | 0.551 / 0.457 / 0.472 | 0.812 / 0.779 / 0.702 |
| External MSD (new hospital, 5-fold ensemble) | 0.555 | 0.823 [0.764, 0.884] (p_max) |

Confound tax (in-distribution): +0.028 AUROC [+0.007, +0.051]. Loose crop: CV Dice ~0.33, detection ~0.70.
Dutch metrics on the 1,939 single-examination patients (patient-disjoint; `tools/eval_cohort.py`).

### Limitations

- trained on crops from **provided** masks; deployment with a predicted mask was measured (Dutch, out-of-fold): AUROC 0.775-0.783 vs 0.787 with a segmenter that never fails, but 0.700 with TotalSegmentator `roi_subset pancreas` (6 % empty masks), so the stage-1 segmenter and its settings must be validated per site
- a small, measurable share of performance comes from the scanner/case-mix confound (bounded ≈0.05 AUROC)
- decision thresholds: p_max operating points set on Dutch CV transfer to MSD/NIH; cc_psz thresholds shift (score-scale effect), so the operating score must be fixed from Dutch CV
- a single-stage reference model, below the PANORAMA challenge winner (PanDx, AUROC 0.926)
- one LOMO training run collapsed to all-background and was re-run (reported)

## PANORAMA 2.5D Classifier (mitigation panel, 2026)

### Model

- architecture: ImageNet ResNet18 on slice triplets (20 depths), attention pooling over depth, auxiliary per-slice lesion loss; scan-level PDAC probability
- input: the tight crops resampled to 1 x 1 x 2 mm, LPS-oriented, 352 x 192 frame, nnU-Net CT normalisation (`tools/panel_build_slices.py`)
- training: tuned ERM (lr 3e-4, 8 epochs, chosen on CV fold 0 inner validation), last-3-epoch soup; 5 CV + 3 LOMO folds x 3 seeds (`tools/panel_train.py`); arms ERM / group-balanced / DFR / GRL / MAE
- checkpoints: local `data/panel_runs/` (not tracked)

### Performance (ERM, 3 seeds)

| Setting | Detection AUROC |
|---|---|
| Dutch CV (seed-averaged scores) | 0.928 (per-seed mean 0.920) |
| LOMO Siemens / Toshiba / Philips held out | 0.885 / 0.837 / 0.670 |
| External MSD+NIH / MSD | 0.884 / 0.824 |
| Calibration (CV) | Brier 0.098, ECE 0.037 |

Confound tax +0.056 (twice the 3D model's; Dutch metrics patient-disjoint); the gain over the 3D model is in-distribution only (ties it on MSD).

### Limitations

- leans on the scanner confound more than the 3D model and loses accuracy on unseen manufacturers; no debiasing arm fixed this
- trained on crops from provided masks; Philips (75 % PDAC) and patients >= 70 are the weakest subgroups

## Thesis-Era CT Model (2025)

### Model

- architecture: ResNet50-based classifier
- input: processed CT slice images after bias-aware preprocessing, orientation correction, segmentation, and cropping
- task: cancer vs control classification

### Interpretation

- the tracked AUC (0.9999 slice / 1.000 patient) is a **dataset-of-origin artifact**: cancer scans came from one source dataset and controls from another, so separating the datasets separates the classes
- ResNet50 embeddings still cluster by source after pixel standardisation
- kept as the motivating example (positive control) for the Q1 work, never as a performance claim
- 2026-10 control: our 2.5D recipe retrained on the same two-source set also reaches 0.996 (vs 0.745 on clinical data at the same training size), and crop statistics alone separate the two collections (0.872): the inflation is in the data, not the old model (`reports/thesis_control/`)

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

- reproduces the published panel and uses a compact, interpretable feature set
- real signal survives within site (BPTB 0.847) and leave-one-site-out (0.78-0.84)

### Limitations

- the cohort is site-confounded: all healthy controls come from one site (BPTB); site alone predicts PDAC at AUROC 0.68 and the panel carries a site tax of +0.062 [+0.027, +0.098] (`reports/biomarker_confound.csv`)
- performance should not be generalized beyond the study setting without external validation

## Fusion Models

### Covered Strategies

- decision-level weighted fusion
- feature-level embedding fusion
- label-matched and label-mismatch sanity controls

### Interpretation

- fusion experiments are methodologically useful
- current fusion results are exploratory only
- they should not be described as evidence of true multimodal clinical benefit because the modalities are not patient-paired
- 2026-10: the pseudo-paired fusion gain equals what a simulation gives under conditional independence of the two tests; with correlated errors it vanishes (`reports/fusion/`)

## Training And Runtime Context

- dependency manager: `uv` (project `.venv`); nnU-Net runs in a separate env, `data/envs/nnunet` (torch 2.8)
- PANORAMA models: `scripts/runpod/` + `tools/` (see `docs/modeling_pipeline.md`)
- thesis orchestration surface: `notebooks/01_multimodal_cancer_detection.ipynb`
- reusable implementation surface: `src/`
- tracked outputs: `reports/` and `figures/`
- local-only assets: `data/`, `models/`, `embeddings/`, `thesis/`

## Current Best Reading Of Results

- PANORAMA 3D CT models: generalise across manufacturers and to an external hospital; the scanner confound is only weakly exploited
- PANORAMA 2.5D classifier: stronger in-distribution, uses the confound more, ties the 3D model externally; debiasing does not beat tuned ERM
- biomarker branch: real signal plus a site confound of the same size as the CT one
- thesis CT model: confounded, kept as the motivating example
- fusion branch: exploratory and hypothesis-generating rather than clinically validated

## Future Model Directions

- uncertainty / OOD flagging (optional)
- nnDetection arm without provided masks (optional robustness control)
- true paired multimodal evaluation if paired cohorts become available
