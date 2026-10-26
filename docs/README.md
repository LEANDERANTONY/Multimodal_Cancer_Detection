# Documentation index

Where to find what. Last reviewed 2026-10-26 (all Q1 experiments complete; write-up next). Start with the **Q1 paper** block if you are picking up the
current work; the **thesis-era** block documents the original 2D pipeline.

## Start here
| Doc | What it answers |
|---|---|
| [`../README.md`](../README.md) | What the project is, current status in one screen |
| [`q1_readiness_and_gaps.md`](q1_readiness_and_gaps.md) | **The living Q1 plan:** findings so far, novelty positioning, gap list, experiment matrix, sequence |
| [`orientation_investigation.md`](orientation_investigation.md) | 155 Dutch PANORAMA scans stored head-to-foot relative to their header: evidence chain (provenance, anatomy, header fields, data-collection stream), effect on TotalSegmentator and the 2.5D classifier |
| [`journal_requirements.md`](journal_requirements.md) | Paper 1 submission requirements: MedIA (target; incl. the 18-question Statement of Significance with planned answers) and Radiology: AI (fallback), each against draft v6 |
| [`literature_novelty_check.md`](literature_novelty_check.md) | Pre-write-up literature search (2026-10-25): verdict per contribution (novel / partly known / supporting), closest prior work, positioning, sources |
| [`data_layout.md`](data_layout.md) | Where every dataset, model checkpoint and result lives locally, the RunPod volume ↔ local map, and **where new files go** |
| [`../reports/README.md`](../reports/README.md) | What each results folder holds |

## Q1 paper (PANORAMA, 3D nnU-Net, scanner confound)
| Doc | What it answers |
|---|---|
| [`modeling_pipeline.md`](modeling_pipeline.md) | How the 3D nnU-Net models were built and evaluated, step by step, with all results |
| [`deployment_and_mitigation_design.md`](deployment_and_mitigation_design.md) | **Pre-registered** design and **every Q1 result**: status section at the top (deployment-ROI, mitigation panel, statistics, biomarker audit, fusion, thesis-cohort control) |
| [`../reports/panorama_confound_audit.md`](../reports/panorama_confound_audit.md) | The dataset audit: scanner confound (Cramér's V 0.44, scanner-only AUROC 0.70), `level` leakage |
| [`architectural_decision_records/`](architectural_decision_records/README.md) | Why the big choices were made (ADR-003 PANORAMA pipeline, ADR-004 modelling/evaluation, ADR-005 pre-registration + outcomes) |
| [`../reports/nnunet_summaries/`](../reports/nnunet_summaries/) | Raw evidence: per-run `summary.json`, per-case detection scores, confound-battery outputs, run logs |
| `preprocessing_audit.md` (local only, git-ignored) | Full CT preprocessing rationale and the v2 (PANORAMA) reprocessing spec |

## Project-level
| Doc | What it answers |
|---|---|
| [`../ROADMAP.md`](../ROADMAP.md) | Publication plan, validation-dataset notes, reviewer-expected analyses, resolved decisions |
| [`../DEVLOG.md`](../DEVLOG.md) | What was done, phase by phase (thesis 2025 → PANORAMA 2026) |
| [`../DEAD_ENDS.md`](../DEAD_ENDS.md) | Failure ledger: what was tried or ruled out, the evidence, and when to revisit; process mistakes and their guards. Read before any experiment |
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
| `tools/loose_cv_reinfer.py`, `tools/confound_tax_ci.py`, `tools/external_threshold_check.py` | Local CV re-inference; confound tax with bootstrap CI; external operating-point transfer |
| `tools/stage1_segment.py`, `tools/stage1_quality.py`, `tools/deploy_infer.py`, `tools/deploy_gap.py`, `tools/dose_response.py`, `tools/plot_dose_response.py`, `tools/deploy_infer_oof.py`, `tools/deploy_identity_gate.py`, `tools/deploy_gap_dutch.py`, `tools/totalseg_failure_analysis.py`, `tools/totalseg_failure_model.py`, `tools/totalseg_organ_probe.py`, `tools/plot_deployment_arms.py` | Deployment-ROI stage 1: TotalSegmentator / official PANORAMA baseline (out-of-fold) pancreas masks on the raw scans; their Dice, crop geometry and lesion containment vs the reference; tight-model detection on the predicted crops; dose-response of detection vs deliberately degraded crops; Dutch out-of-fold stage 2, its identity gate, and the Dutch oracle-vs-predicted gap (confound tax under predicted crops); why TotalSegmentator fails (scan features, scanner- and label-matched model) |
| `tools/panel_build_slices.py`, `tools/panel_train.py`, `tools/panel_queue.py` (queues in `configs/panel/`), `tools/panel_eval.py`, `tools/panel_dfr.py`, `tools/panel_mae.py`, `tools/panel_planted_eval.py`, `tools/panel_stats.py`, `tools/plot_panel.py` | Mitigation panel: 2.5D slice dataset from the tight crops (LPS, 1 x 1 x 2 mm, 352 x 192, per-slice lesion counts); 2.5D classifier with ERM / group-balanced / GRL arms over CV or LOMO, inner-validation selection, last-epoch soup, per-case scores; resumable one-GPU-job queue; evaluator (CV tax with CI, LOMO, external); DFR (last layer on group-balanced held-out data); MAE (masked-image pretraining, then ERM); planted-shortcut control (clean vs flipped marker); arm-vs-ERM statistics and the panel figure |
| `tools/final_stats.py` | Paper statistics: 3D vs 2.5D DeLong, calibration, subgroups, mask-provenance leakage check, 3D Dice and LOMO |
| `tools/eval_cohort.py` | Patient-disjoint evaluation set (1,939 single-examination Dutch patients), imported by every Dutch metric script |
| `tools/biomarker_confound.py`, `tools/fusion_sim.py`, `tools/plot_fusion.py` | The confound audit on the urinary biomarker cohort; fusion of the unpaired cohorts as a dependence-sensitivity simulation plus the pseudo-pairing demonstration, and its figure |
| `tools/plot_paper_figures.py` | Manuscript figures: Fig. 1 study design and the supplementary figures (TotalSegmentator failure geometry, subgroup AUROCs, calibration, thesis case study) |
| `tools/thesis_control.py`, `tools/thesis_control_eval.py`, `tools/thesis_retrain.py` | Thesis two-source set through the PANORAMA deployment pipeline; our recipe retrained on it, cross-tested on PANORAMA, size-matched control, crop-statistics source probe |
