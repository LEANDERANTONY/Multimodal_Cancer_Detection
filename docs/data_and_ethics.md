# Data And Ethics

This project uses medical imaging and biomarker data related to pancreatic cancer detection. Because the work sits in a health context, the main ethical obligation is not just model quality, but careful handling of privacy, bias, uncertainty, and claims.

## Data Handling

- raw clinical data is intentionally kept local and is not tracked in Git
- processed datasets are also kept local because they remain research assets, not public benchmark files
- thesis files, checkpoints, embeddings, and presentation materials are treated as local-only by default
- contributors should never commit patient-level source data, exported scans, or derived files that could create privacy or governance issues

## Privacy Posture

- this repository is structured to avoid publishing raw patient data
- tracked artifacts are limited to lightweight code, documentation, reports, and curated figures
- any future sharing of sample data should use explicitly approved, de-identified, non-sensitive examples

## Bias And Scientific Validity

Bias is a central concern in this project, especially for the CT branch.

The dissertation-aligned interpretation is:

- CT results are strong, but they may still contain residual dataset-of-origin signal
- the biomarker branch is the cleanest reproducible result in the repository
- fusion is exploratory because CT and biomarker cohorts are not patient-paired

That means high headline metrics should not be read as proof of clinical readiness.

## Intended Use

This repository is intended for:

- research documentation
- thesis support
- method development
- portfolio demonstration of multimodal and bias-aware ML work

This repository is not intended for:

- clinical decision support
- patient triage
- diagnosis in real care settings
- unattended deployment in healthcare environments

## Ethical Reporting Expectations

When describing the project publicly, keep these points explicit:

- the CT branch required extensive bias-aware preprocessing and still has unresolved domain-generalization questions
- the biomarker branch is more defensible than the fusion branch as a standalone positive result
- decision-level and feature-level fusion were evaluated under synthetic pairing assumptions, not true paired-patient multimodal data
- the project is a research artifact, not a validated clinical system

## Future Ethical And Methodological Improvements

- external validation on additional CT cohorts
- domain-adversarial CT training to suppress dataset-of-origin shortcuts
- clearer uncertainty reporting and calibration tracking
- real paired multimodal cohorts instead of synthetic pairing
- stronger dataset documentation and governance notes if data-sharing constraints change
