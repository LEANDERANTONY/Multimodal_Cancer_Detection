# Local data, model and results layout

Where everything lives on the workstation (`D:/Documents/Projects/Multimodal_Cancer_Detection`).
`data/` and `models/` are git-ignored (local only); `reports/` holds tracked lightweight results.
Everything needed to work offline is local — the RunPod volume `panaroma_roi` is only a cloud mirror.
Pipeline that produced it: `docs/modeling_pipeline.md`.

## Datasets — `data/`
| Path | What |
|---|---|
| `data/raw/ct/panorama/` | PANORAMA raw CT (2238 studies, ~182 GB) |
| `data/raw/ct/panorama_labels/` | PANORAMA masks (`automatic_labels/`, `manual_labels/`) + `clinical_information.xlsx`. Mask legend: 1 lesion, 2 veins, 3 arteries, 4 pancreas, 5 duct, 6 CBD |
| `data/raw/ct/cancer/`, `data/raw/ct/control/` | Thesis-era TCIA cohorts (Pancreatic-CT-CBCT-SEG cancer, NIH Pancreas-CT controls) |
| `data/raw/biomarkers/` | Urinary biomarker data |
| `data/processed/ct/nnunet_raw/Dataset700_PanoramaPDAC/` | **Loose ROI** nnU-Net raw (margin 150×100×40 mm): 1964 Dutch `imagesTr/labelsTr` + 274 MSD/NIH `imagesTs/labelsTs`, `build_manifest.csv`, `splits_final.json`. `panorama_roi_meta/` = manifest/log/splits of the first crop pass |
| `data/processed/ct/nnunet_raw/Dataset701_PanoramaPDAC_tight/` | **Tight ROI** nnU-Net raw (margin 100×50×15 mm): all 2238 in `imagesTr` (training strips to the 1964 Dutch IDs) + `roi_build_qc.csv` |

No local tars are kept (deleted 2026-10-05 as duplicates of the folders above). The volume holds
`Dataset701_tight.tar`; to upload a dataset again, re-create its tar first, e.g.
`tar cf Dataset701_tight.tar -C data/processed/ct/nnunet_raw Dataset701_PanoramaPDAC_tight`.
The first loose crop pass (`data/processed/ct/panorama_roi/`) was deleted too: same voxels as Dataset700 but with the image origin zeroed.
| `data/processed/ct_*`, `*.csv` | Thesis-era 2D pipeline outputs |
| `data/envs/nnunet/` | Separate Python env for nnU-Net (torch 2.8.0+cu128 + nnunetv2). nnU-Net excludes torch 2.9, which the project `.venv` pins — run nnU-Net code with `data/envs/nnunet/Scripts/python.exe` |

## Trained nnU-Net models — `models/nnunet/`
Each run folder is a valid `nnUNet_results` root (set `nnUNet_results=<run folder>`).
All 3d_fullres, `nnUNetTrainer_250epochs`, `checkpoint_final.pth` + `checkpoint_best.pth` per fold, plus
`plans.json`, `dataset.json`, training logs, `progress.png`, and `validation/` (predicted masks + `summary.json`; softmax `.npz` not kept).

| Run folder | Dataset | Folds |
|---|---|---|
| `models/nnunet/loose_cv/` | Dataset700 | 5-fold CV (random) |
| `models/nnunet/loose_lomo/` | Dataset700 | LOMO: fold 0 Siemens, 1 Toshiba, 2 Philips held out |
| `models/nnunet/tight_cv/` | Dataset701 | 5-fold CV (random) |
| `models/nnunet/tight_lomo/` | Dataset701 | LOMO as above; `fold_2_collapsed/` = failed first fold-2 run (logs only) |
| `models/nnunet/runpod_volume_misc/` | — | Everything else from the volume: run logs, drivers as they ran, `splits_final_lomo.json` (manufacturer-holdout splits), feature npz, `tight_battery/` |

`models/*.pt` (top level) are the thesis-era 2D ResNet/biomarker checkpoints.

## Results — `reports/`
| Path | What |
|---|---|
| `reports/nnunet_summaries/<volume run name>/` | `summary.json`, plans, logs per run (`nnunet_results` = loose CV, `_lomo`, `_tight`, `_tight_lomo`) |
| `reports/nnunet_summaries/tight_battery/` | Tight confound battery: `cv_scores.csv`, `lomo_scores.csv` (per-case p_max/p_sum/cc_psz), `feature_diag_tight.npz` |
| `reports/nnunet_summaries/loose_battery/` | Loose re-inferred CV per-case scores (`tools/loose_cv_reinfer.py`) |
| `reports/nnunet_summaries/feature_diag_features.npz` | Loose feature-probe features |
| `reports/panorama_confound_audit.md` | Dataset confound audit (scanner-only AUROC etc.) |

## Volume ↔ local name map
| RunPod volume (`/workspace`) | Local |
|---|---|
| `nnunet_results` | `models/nnunet/loose_cv` |
| `nnunet_results_lomo` | `models/nnunet/loose_lomo` |
| `nnunet_results_tight` | `models/nnunet/tight_cv` |
| `nnunet_results_tight_lomo` | `models/nnunet/tight_lomo` |
| `Dataset701_tight.tar` | `data/processed/ct/nnunet_raw/Dataset701_PanoramaPDAC_tight/` (extracted; no local tar) |
| other files | `models/nnunet/runpod_volume_misc/` |
