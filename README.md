# Multimodal Pancreatic Cancer Detection

This repository is the GitHub-tracked working project for a pancreatic cancer detection study that combines CT imaging and urine biomarker modelling. The project is currently in a hybrid state:

- The modular repo scaffold is preserved in `src/`, `configs/`, `docs/`, and `results/`.
- The latest end-to-end experimentation lives in `notebooks/01_multimodal_cancer_detection.ipynb`.
- Curated reports and publication-ready figures are tracked in `reports/` and `figures/`.
- Raw data, processed datasets, model weights, embeddings, and thesis assets remain local-only and are intentionally ignored by Git.

The immediate next step after stabilizing this snapshot is to refactor reusable notebook logic back into `src/` so the notebook becomes orchestration and analysis rather than the only implementation surface.

## Repository Status

This repo is being actively consolidated from a newer local working copy. Expect a mix of:

- reusable preprocessing code under `src/data/preprocess/`
- analysis outputs and final summaries under `reports/`
- curated visual artifacts under `figures/`
- one primary notebook that captures the latest full experimental flow

If you are looking for the most up-to-date experiment narrative, start with:

- `notebooks/01_multimodal_cancer_detection.ipynb`
- `reports/final_summary.json`
- `reports/final_model_comparison.csv`

## Current Pipeline Snapshot

The main notebook currently covers:

1. Setup, data loading, and preprocessing
2. Dataset bias checks and unsupervised structure analysis
3. CT classification experiments
4. Biomarker MLP experiments
5. Exploratory multimodal fusion
6. Final summary and export

The tracked preprocessing scripts currently available in the modular codebase are:

- `src/data/preprocess/ct_preprocess.py`
- `src/data/preprocess/biomarker_preprocess.py`

## Result Snapshot

The latest exported summary in `reports/final_summary.json` reports:

- CT model: ResNet50 (global)
- CT slice-level AUC: 0.9999
- CT patient-level AUC: 1.0000
- Biomarker model: MLP (64-32)
- Biomarker test AUC: 0.9439
- Fusion note: fusion experiments are exploratory because synthetic pairing was used across different patient cohorts

These numbers should be interpreted as experiment outputs from the current working pipeline, not as clinically validated deployment metrics.

## Project Layout

```text
Multimodal_Cancer_Detection/
|-- configs/
|   `-- paths.yaml
|-- data/
|   |-- .gitkeep
|   `-- raw/
|       `-- .gitkeep
|-- docs/
|   |-- architectural_decision_records/
|   |-- figures-guide.md
|   |-- LICENSE_Pancreatic_CT_Cancer.txt
|   |-- LICENSE_Pancreatic_CT_Control.txt
|   `-- timeline.md
|-- figures/
|   `-- curated tracked PNG outputs
|-- notebooks/
|   `-- 01_multimodal_cancer_detection.ipynb
|-- reports/
|   `-- curated tracked CSV and JSON outputs
|-- results/
|-- src/
|   |-- cli/
|   |-- data/
|   |   `-- preprocess/
|   |-- fusion/
|   |-- interpretability/
|   |-- models/
|   |-- uncertainty/
|   `-- utils/
|-- tools/
|   `-- helper scripts for notebook/report maintenance
|-- requirements.txt
`-- README.md
```

## Local-Only Assets

The following paths are intentionally ignored and should stay local unless there is a deliberate change in repo policy:

- `data/raw/`
- `data/processed/`
- `models/`
- `embeddings/`
- `thesis/`
- local archive/video files
- temporary staging or Drive-sync folders

That split keeps the GitHub repo focused on code, lightweight configuration, curated outputs, and documentation.

## Setup

Create a virtual environment and install the current baseline dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: `requirements.txt` is a baseline file and is expected to be refreshed after the repo refactor and environment rebuild.

## Data Paths

The current path configuration is in `configs/paths.yaml`:

```yaml
raw_ct_dir: data/raw/ct
processed_ct_dir: data/processed/ct
raw_biomarkers_dir: data/raw/biomarkers
processed_biomarkers_dir: data/processed/biomarkers
figures_dir: figures
```

Place local datasets under the ignored `data/raw/` tree using those conventions.

## Running What Exists Today

### Preprocessing scripts

```powershell
python src/data/preprocess/ct_preprocess.py
python src/data/preprocess/biomarker_preprocess.py
```

### Main experiment notebook

Open and run:

```text
notebooks/01_multimodal_cancer_detection.ipynb
```

This notebook is currently the main source of truth for the complete multimodal workflow.

## Tracked Outputs

Examples of tracked artifacts already in the repo:

- CT model summaries and predictions in `reports/ct_*.json` and `reports/ct_*.csv`
- fusion experiment summaries in `reports/fusion_*.csv` and `reports/fusion_*.json`
- final comparison tables in `reports/final_model_comparison.csv` and `reports/final_summary.json`
- curated plots and Grad-CAM style outputs in `figures/`

## Documentation

Supporting project documentation lives in:

- `docs/timeline.md`
- `docs/architectural_decision_records/ADR-001.md`
- `docs/architectural_decision_records/ADR-002.md`

Some documentation files are placeholders and will be filled in as the consolidation continues.

## Known Gaps

- The repo is mid-consolidation from a newer local working copy.
- The main notebook still contains logic that should be moved into `src/`.
- `requirements.txt` has not yet been rebuilt from the final environment.
- Some docs and helper files are placeholders and need completion.
- Curated figures and reports are tracked, but bulk intermediate artifacts remain intentionally local.

## Next Planned Cleanup

1. Finalize the first consolidation commit.
2. Push the stabilized hybrid snapshot to GitHub.
3. Refactor reusable notebook code into modules under `src/`.
4. Rebuild `.venv` in this repo and refresh `requirements.txt`.
5. Tighten README usage examples around the refactored module entry points.

## License

See `LICENSE` for the repository license and `docs/` for dataset-specific license notes currently tracked with the project.
