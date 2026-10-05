# Documentation index

Where to find what. Last reviewed 2026-10-05. Start with the **Q1 paper** block if you are picking up the
current work; the **thesis-era** block documents the original 2D pipeline.

## Start here
| Doc | What it answers |
|---|---|
| [`../README.md`](../README.md) | What the project is, current status in one screen |
| [`q1_readiness_and_gaps.md`](q1_readiness_and_gaps.md) | **The living Q1 plan:** findings so far, novelty positioning, gap list, experiment matrix, sequence |
| [`data_layout.md`](data_layout.md) | Where every dataset, model checkpoint and result lives locally, the RunPod volume ↔ local map, and **where new files go** |
| [`../reports/README.md`](../reports/README.md) | What each results folder holds |

## Q1 paper (PANORAMA, 3D nnU-Net, scanner confound)
| Doc | What it answers |
|---|---|
| [`modeling_pipeline.md`](modeling_pipeline.md) | How the 3D nnU-Net models were built and evaluated, step by step, with all results |
| [`deployment_and_mitigation_design.md`](deployment_and_mitigation_design.md) | **Pre-registered** design of the deployment-ROI experiment and the mitigation panel |
| [`../reports/panorama_confound_audit.md`](../reports/panorama_confound_audit.md) | The dataset audit: scanner confound (Cramér's V 0.44, scanner-only AUROC 0.70), `level` leakage |
| [`architectural_decision_records/`](architectural_decision_records/README.md) | Why the big choices were made (ADR-003 PANORAMA pipeline, ADR-004 modelling/evaluation, ADR-005 pre-registration) |
| [`../reports/nnunet_summaries/`](../reports/nnunet_summaries/) | Raw evidence: per-run `summary.json`, per-case detection scores, confound-battery outputs, run logs |
| `preprocessing_audit.md` (local only, git-ignored) | Full CT preprocessing rationale and the v2 (PANORAMA) reprocessing spec |

## Project-level
| Doc | What it answers |
|---|---|
| [`../ROADMAP.md`](../ROADMAP.md) | Publication plan, validation-dataset notes, reviewer-expected analyses, resolved decisions |
| [`../DEVLOG.md`](../DEVLOG.md) | What was done, phase by phase (thesis 2025 → PANORAMA 2026) |
| [`project_strategy.md`](project_strategy.md) | Scope boundaries and why the repo is shaped as it is |
| [`architecture.md`](architecture.md) | Code map: thesis modules in `src/`, PANORAMA pipeline in `scripts/runpod/` + `tools/` |
| [`model_card.md`](model_card.md) | Every model, what it is, how good it is, and its limitations |
| [`data_and_ethics.md`](data_and_ethics.md) | Data licences, local-only policy, intended use |
| [`quickstart.md`](quickstart.md) | Setup (incl. the separate nnU-Net env) and reading order |
| [`figures-guide.md`](figures-guide.md) | What belongs in `figures/` |
| [`../folder_structure.txt`](../folder_structure.txt) | Repository tree at a glance |

## Code entry points for the Q1 work
| Path | What |
|---|---|
| `scripts/runpod/` | Pod-side drivers (training, LOMO, confound battery, detection scoring, feature probe, external validation, upload/download) |
| `tools/build_roi_dataset.py` | ROI crop builder (loose 150×100×40 mm / tight 100×50×15 mm) |
| `tools/scanner_only_shortcut.py` | Scanner-metadata-only AUROC baseline |
| `scripts/runpod/detection_candidate.py`, `scripts/runpod/feature_diag.py` | Detection scoring from softmax; encoder feature probe (pod-side) |
| `tools/loose_cv_reinfer.py`, `tools/confound_tax_ci.py` | Local CV re-inference; confound tax with bootstrap CI |
