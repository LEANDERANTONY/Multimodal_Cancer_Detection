# Quickstart

This project is maintained as a research repository, not a packaged application. The fastest way to get oriented is to separate what works without local data from what requires your private processed assets.

## 1. Set up the environment

```powershell
uv sync
```

This uses:

- `pyproject.toml`
- `uv.lock`
- `.python-version`

## 2. Run the lightweight validation path

These checks do not require the local CT/biomarker datasets.

```powershell
uv run python -m compileall src
uv run pytest
```

That validates the reusable helper layers, fusion utilities, and report-summary code paths tracked in Git.

## 3. Review the tracked outputs first

If you want the fastest understanding of the project without touching local data:

- inspect `reports/final_summary.json`
- inspect `reports/model_comparison.csv`
- inspect the curated figures under `figures/`
- read `docs/architecture.md` and `docs/model_card.md`

## 4. Run the main notebook with local processed data

Open:

```text
notebooks/01_multimodal_cancer_detection.ipynb
```

The maintained notebook flow is processed-data-first. In normal use it expects local artifacts under:

- `data/processed/`
- `reports/`
- `models/` for local checkpoints when relevant

The notebook should not need raw data for ordinary analysis runs.

## 5. Rebuild processed data only when needed

Use preprocessing scripts only if you are intentionally regenerating processed assets from local raw data:

```powershell
uv run python src/data/preprocess/ct_preprocess.py
uv run python src/data/preprocess/biomarker_preprocess.py
```

Those scripts are rebuild steps, not required for everyday notebook execution.

## 6. Expected local-only assets

These stay outside Git and are expected to exist only on local machines with approved access:

- `data/raw/`
- `data/processed/`
- `models/`
- `embeddings/`
- `thesis/`

## 7. Practical reading order

If you are new to the repo, use this order:

1. `README.md`
2. `docs/quickstart.md`
3. `docs/architecture.md`
4. `docs/data_and_ethics.md`
5. `docs/model_card.md`
6. `notebooks/01_multimodal_cancer_detection.ipynb`
