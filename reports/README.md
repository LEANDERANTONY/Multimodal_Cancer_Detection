# reports/

Tracked, lightweight results (CSV / JSON / Markdown / small npz). No images or scans.

| Path | What |
|---|---|
| `*.csv`, `*.json` (flat, this folder) | Thesis-era (2025) outputs, read and written by the notebook and `src/` — keep flat, do not move |
| `panorama_confound_audit.md` | PANORAMA dataset confound audit: scanner ↔ label (Cramér's V 0.44, scanner-only AUROC 0.70), `level` near-label leakage |
| `nnunet_summaries/` | Evidence for the Q1 3D nnU-Net work (below) |

## nnunet_summaries/
| Path | What |
|---|---|
| `nnunet_results/` | Loose ROI, 5-fold CV — per-fold `validation/summary.json`, plans |
| `nnunet_results_lomo/` | Loose ROI, LOMO (fold 0 Siemens, 1 Toshiba, 2 Philips held out) |
| `nnunet_results_tight/` | Tight ROI, 5-fold CV |
| `nnunet_results_tight_lomo/` | Tight ROI, LOMO; incl. training logs and `fold_2_collapsed/` (failed first fold-2 run) |
| `tight_battery/` | Tight confound battery: per-case scores `cv_scores.csv`, `lomo_scores.csv`; `feature_diag_tight.npz` |
| `loose_battery/` | Loose re-inferred CV per-case scores `cv_scores.csv`; `feature_diag_features.npz` |
| `external/` | External validation (MSD + NIH): `external_cases.csv`, per-case scores `tight_external.csv` / `loose_external.csv` (5-fold ensembles), `tight_external_fold{0-4}.csv` (single folds), `threshold_check.csv` (operating-point transfer) |
| `deployment_roi/` | Deployment-ROI experiment: `stage1_quality_<arm>_<cohort>.csv` (per-case segmentation Dice vs reference, centroid offset, tight-crop IoU with the oracle crop, lesion containment; `tools/stage1_quality.py`), `detect_<arm>_external.csv` (per-case detection scores on predicted-mask crops; `tools/deploy_infer.py`), `deploy_gap_external.csv` (oracle vs predicted, paired bootstrap; `tools/deploy_gap.py`), `dose_response_msd.csv` / `dose_response_summary.csv` (detection vs deliberately shifted / rescaled oracle crops; `tools/dose_response.py`, `tools/plot_dose_response.py`) |
| `run_logs/` | Pod run logs |

Folder names follow the RunPod volume run names; the matching local checkpoints are in
`models/nnunet/{loose_cv, loose_lomo, tight_cv, tight_lomo}` (map: `docs/data_layout.md`).
Per-case CSV columns: `fold, case, scanner, y, p_max, p_sum, cc_psz` (external files add `source, n_ref, n_pred, dice`).
Analysis from these files: `tools/confound_tax_ci.py` (confound tax + bootstrap CI), `tools/external_threshold_check.py` (operating points).
