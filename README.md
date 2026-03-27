# Multimodal Pancreatic Cancer Detection

[![CI](https://github.com/LEANDERANTONY/Multimodal_Cancer_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/LEANDERANTONY/Multimodal_Cancer_Detection/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Multimodal Pancreatic Cancer Detection is a bias-aware research repository for pancreatic cancer detection using CT imaging and urinary biomarkers. The implemented workflow combines bias-aware CT preprocessing, ResNet50-based CT classification, a seven-feature biomarker MLP, and exploratory multimodal fusion under synthetic pairing constraints.

## What It Does

- detects and mitigates cross-dataset shortcut risk in CT slices before CT model training
- trains a ResNet50 CT classifier with slice-level and patient-level evaluation
- trains a urinary biomarker MLP on the seven-feature panel used in the dissertation
- evaluates decision-level and feature-level fusion with multi-seed repeats and label-mismatch controls
- exports tracked summaries, comparison tables, and curated figures for thesis and research reporting

## Research Flow

1. Prepare CT and biomarker inputs with the preprocessing scripts when fresh local preprocessing is needed.
2. Run CT bias checks and iterative mitigation.
3. Apply orientation correction, body segmentation, and cropping for CT slices.
4. Train and evaluate the CT ResNet50 model.
5. Train and evaluate the biomarker MLP.
6. Run exploratory fusion experiments with negative controls.
7. Export final summaries, comparison tables, and figures.

## Visual Snapshot

### Bias Mitigation Diagnostic

![Dataset bias check](figures/dataset_bias_check.png)

### Comparative Model Summary

![Final model comparison](figures/final_model_comparison.png)

## Result Snapshot

The latest tracked summary in `reports/final_summary.json` reports:

- CT model: ResNet50 (global)
- CT slice-level AUC: 0.9999
- CT patient-level AUC: 1.0000
- Biomarker model: MLP (64-32)
- Biomarker test AUC: 0.9439

Those numbers should be read carefully:

- the biomarker branch is the clearest reproducible positive result in the project
- the CT branch is strong but still scientifically ambiguous because residual domain structure may persist in deep features
- fusion remains exploratory because CT and biomarker cohorts are not patient-paired

## Documentation

Supporting project documentation lives in:

- `docs/quickstart.md`
- `docs/architecture.md`
- `docs/data_and_ethics.md`
- `docs/model_card.md`
- `ROADMAP.md`
- `DEVLOG.md`
- `docs/timeline.md`
- `docs/figures-guide.md`
- `docs/architectural_decision_records/README.md`
- `docs/architectural_decision_records/ADR-001.md`
- `docs/architectural_decision_records/ADR-002.md`
