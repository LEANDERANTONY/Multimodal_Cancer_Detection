# Local data, model and results layout

Where everything lives on the workstation (`D:/Documents/Projects/Multimodal_Cancer_Detection`), and
**where new things go** (last section). `data/` and `models/` are git-ignored (local only); `reports/`
and `figures/` hold tracked lightweight outputs. Everything needed to work offline is local — the
RunPod volume `panaroma_roi` is only a cloud mirror. Pipeline that produced it: `docs/modeling_pipeline.md`.

## Datasets — `data/` (local only)
| Path | What |
|---|---|
| `data/raw/ct/panorama/` | PANORAMA raw CT (2238 studies, ~182 GB) |
| `data/raw/ct/panorama_labels/` | PANORAMA masks (`automatic_labels/`, `manual_labels/`) + `clinical_information.xlsx`. Mask legend: 1 lesion, 2 veins, 3 arteries, 4 pancreas, 5 duct, 6 CBD |
| `data/raw/ct/cancer/`, `data/raw/ct/control/` | Thesis-era TCIA cohorts (Pancreatic-CT-CBCT-SEG cancer, NIH Pancreas-CT controls) |
| `data/raw/biomarkers/` | Urinary biomarker data |
| `data/processed/ct/nnunet_raw/Dataset700_PanoramaPDAC/` | **Loose ROI** nnU-Net raw (margin 150×100×40 mm): 1964 Dutch `imagesTr/labelsTr` + 274 MSD/NIH `imagesTs/labelsTs`, `build_manifest.csv` (case → source), `splits_final.json`; `panorama_roi_meta/` = log/manifest/splits of the first crop pass |
| `data/processed/ct/nnunet_raw/Dataset701_PanoramaPDAC_tight/` | **Tight ROI** nnU-Net raw (margin 100×50×15 mm): all 2238 in `imagesTr` (training strips to the 1964 Dutch IDs) + `roi_build_qc.csv` |
| `data/processed/ct_*/`, `data/processed/*.csv` | Thesis-era 2D pipeline outputs (oriented / segmented / cropped slices, indices, biomarker CSVs) |
| `data/processed/ct/stage1_masks/` | Deployment-ROI stage-1 pancreas masks on the raw scans: `totalseg/` (arm A), `baseline_oof/` (arm B, official PANORAMA baseline, out-of-fold); `_smoke_totalseg/` = 1-case install test |
| `data/processed/ct/deploy_crops/<arm>/` | Tight crops cut from each arm's predicted masks (`imagesTs`, `labelsTs` = reference lesion) and the tight-ensemble predictions (`pred/`); `tools/deploy_infer.py` |
| `data/envs/totalseg/` | TotalSegmentator 2.18 weights (`TOTALSEG_HOME_DIR`); the package itself is installed in `data/envs/nnunet` |
| `data/envs/nnunet/` | Separate Python env for nnU-Net (torch 2.8.0+cu128 + nnunetv2). nnU-Net excludes torch 2.9, which the project `.venv` pins — run nnU-Net code with `data/envs/nnunet/Scripts/python.exe` |

**No local tars.** Tars are only made for uploads and deleted afterwards (all were deleted 2026-10-05).
To upload a dataset again: `tar cf X.tar -C data/processed/ct/nnunet_raw <DatasetFolder>`. The volume
keeps `Dataset701_tight.tar` and `Dataset700_Ts.tar` (the loose external test scans). The first loose
crop pass (`data/processed/ct/panorama_roi/`) was deleted: same voxels as Dataset700, image origin zeroed.

## Trained models — `models/` (local only)
| Path | What |
|---|---|
| `models/nnunet/loose_cv/` | Dataset700, 5-fold CV (random) |
| `models/nnunet/loose_lomo/` | Dataset700, LOMO: fold 0 Siemens, 1 Toshiba, 2 Philips held out |
| `models/nnunet/tight_cv/` | Dataset701, 5-fold CV (random) |
| `models/nnunet/tight_lomo/` | Dataset701, LOMO as above; `fold_2_collapsed/` = failed first fold-2 run (logs only) |
| `models/nnunet/runpod_volume_misc/` | Everything else from the volume: run logs, drivers as they ran, `splits_final_lomo.json` (manufacturer-holdout splits), feature npz, `tight_battery/` |
| `models/panorama_baseline/` | Official PANORAMA baseline pancreas segmenter (Dataset103, nnU-Net v2, 5 folds; Zenodo 11160381, MD5-checked) + its fold file — stage 1, arm B |
| `models/*.pt` | Thesis-era 2D ResNet / biomarker checkpoints |

Each `models/nnunet/<run>/` is a valid `nnUNet_results` root: `Dataset70x_*/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/`
with `plans.json`, `dataset.json`, and per fold `checkpoint_final.pth`, `checkpoint_best.pth`, training
log, `progress.png`, `validation/` (predicted masks + `summary.json`; softmax `.npz` not kept).

## Results — `reports/` (tracked) — see also `reports/README.md`
| Path | What |
|---|---|
| `reports/*.csv`, `reports/*.json` (flat) | Thesis-era outputs, read and written by the notebook and `src/` — do not move |
| `reports/panorama_confound_audit.md` | Dataset confound audit (scanner-only AUROC, `level` leakage) |
| `reports/nnunet_summaries/nnunet_results*/` | Per-run `summary.json`, plans, debug, training logs (volume run names: `nnunet_results` = loose CV, `_lomo`, `_tight`, `_tight_lomo`) |
| `reports/nnunet_summaries/tight_battery/` | Tight confound battery: `cv_scores.csv`, `lomo_scores.csv` (per-case p_max / p_sum / cc_psz), `feature_diag_tight.npz` |
| `reports/nnunet_summaries/loose_battery/` | Loose model: re-inferred CV per-case scores (`cv_scores.csv`), feature-probe features (`feature_diag_features.npz`) |
| `reports/nnunet_summaries/external/` | External validation: `external_cases.csv` (case, source MSD/NIH, label), `tight_external.csv`, `loose_external.csv`, per-fold files |
| `reports/deployment_roi/` | Deployment-ROI experiment: stage-1 segmentation quality and crop geometry per case and arm and detection on the predicted crops vs oracle |
| `reports/nnunet_summaries/run_logs/` | Pod run logs (LOMO, confound battery, fold-2 retry) |

## Figures — `figures/` (tracked)
Flat `figures/*.png` are thesis-era (written by the notebook). Q1 / PANORAMA figures go in `figures/panorama/`.
`figures/local_thesis_audit/` (git-ignored, ~190 MB) holds thesis-era bulk audit images and screenshots
(`visual_audit/`, `body_segmentation_filtered/`, `body_segmentation_examples_filtered.png`, `Screenshot *.png`).

## Thesis-era local folders (local only, unchanged)
`thesis/` (dissertation, interim report, papers), `embeddings/` (ResNet embeddings),
`VideoPresentation_CodeFile_Dataset/` (presentation assets), `notebooks/` (tracked thesis notebook).

## Volume ↔ local name map
| RunPod volume (`/workspace`) | Local |
|---|---|
| `nnunet_results` | `models/nnunet/loose_cv` |
| `nnunet_results_lomo` | `models/nnunet/loose_lomo` |
| `nnunet_results_tight` | `models/nnunet/tight_cv` |
| `nnunet_results_tight_lomo` | `models/nnunet/tight_lomo` |
| `Dataset701_tight.tar` | `data/processed/ct/nnunet_raw/Dataset701_PanoramaPDAC_tight/` |
| `Dataset700_Ts.tar` | `data/processed/ct/nnunet_raw/Dataset700_PanoramaPDAC/imagesTs` + `labelsTs` |
| `ext_eval/` | `reports/nnunet_summaries/external/` |
| other files | `models/nnunet/runpod_volume_misc/` |

## Where new things go (filing rule)
| New thing | Goes in | Then update |
|---|---|---|
| A dataset (crops, predicted masks, nnU-Net raw) | `data/processed/ct/<name>/` (nnU-Net raw: `data/processed/ct/nnunet_raw/Dataset7xx_*`) | this doc |
| A trained model / run | `models/nnunet/<run_name>/` (or `models/<arm>/` for 2.5D panel arms) | this doc, `docs/model_card.md` |
| Per-run summaries, per-case scores, run logs | `reports/nnunet_summaries/<experiment>/` (or `reports/<experiment>/` for non-nnU-Net work) | this doc, `reports/README.md` |
| A paper figure | `figures/panorama/` | `docs/figures-guide.md` if it's a new kind |
| A pod-side driver | `scripts/runpod/` | `docs/architecture.md` table |
| A local analysis script | `tools/` (thesis-writing utilities: `tools/thesis/`) | `docs/README.md` code entry points |
| A new method / experiment doc | `docs/` | `docs/README.md` index (and an ADR if it is a decision) |
| A result | the experiment's doc (`docs/modeling_pipeline.md` or its own doc) | `docs/q1_readiness_and_gaps.md` status, `DEVLOG.md` |
| Temporary files (upload tars, scratch) | delete after use | — |
